"""
FreshSense core analysis module.

What happens when a photo is analysed
-------------------------------------
1. FIND FRUITS   - on a plain background the photo is split into separate objects,
                   so every fruit is judged on its own (find_fruit_regions).
2. IS IT A FRUIT?- each object gets a "fruit score" (the model's strongest class
                   score). If it is too low the object is NOT a fruit FreshSense
                   knows, and we say so instead of guessing (the "gate").
3. JUDGE EACH    - fruit name + freshness score + status, one result per fruit.
                   Mixed photos (one fresh, one rotten) are reported as mixed -
                   never as one overall "fresh".
4. FALLBACK      - if there is no trained model (or it crashes) the OpenCV colour
                   analysis is used so the app still works.

The model is the 16-class MobileNetV2 classifier (8 fruits x fresh/rotten).
"""

import os
import io
import json
import base64
import tempfile
import numpy as np
import cv2
from PIL import Image

import grid_map
import shelf_life

# Some managed Windows profiles deny writes to default Matplotlib/YOLO cache
# folders. Select the writable temp folder before importing ML libraries.
_RUNTIME_CACHE = os.path.join(tempfile.gettempdir(), "freshsense6-yolo-cache")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_RUNTIME_CACHE, "config"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RUNTIME_CACHE, "matplotlib"))

# ---------------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("FRESHSENSE_MODEL_DIR", os.path.join(_HERE, "model"))
_LEGACY_MODEL_PATH = os.environ.get("FRESHSENSE_MODEL_PATH", os.path.join(MODEL_DIR, "freshsense_model.h5"))
_LEGACY_LABELS_PATH = os.environ.get("FRESHSENSE_LABELS_PATH", os.path.join(MODEL_DIR, "labels.txt"))
_KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "fruit_condition.keras")
_METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
_DETECTOR_PATH = os.environ.get(
    "FRESHSENSE_DETECTOR_PATH", os.path.join(MODEL_DIR, "yolo11n.pt")
)
GATE_CONFIG_PATH = os.path.join(MODEL_DIR, "gate_config.json")

# "Is it a fruit?" threshold on the model's strongest class score (max logit).
# During development 14 of 15 real non-fruit photos (astronaut, cat, coffee cup,
# coins, text page, rocket ...) scored below 0.5; the exception was a gravel
# texture that looks like a pile of grapes. The value is a starting point -
# tune it for YOUR photos with calibrate_gate.py.
DEFAULT_MIN_LOGIT = -1.0
DEFAULT_MIN_SOFTMAX = 0.5          # used only if the model's internals are not readable
WEAK_GATE_MARGIN = 1.5             # accepted but within this margin of the threshold -> add a warning

STATUS_THRESHOLDS = {"fresh_min": 70, "moderate_min": 40}
STATUS_BASIS = "experimental_score_band"
RESULT_DISCLAIMER = (
    "Experimental visual estimate based on visible image features. Fresh, Moderate, and "
    "Spoiled statuses are score bands; Moderate is not a class in the current model. "
    "The score thresholds have not been scientifically validated, and this result cannot establish food safety."
)

# If you retrain with an extra "not_fruit" class (see train.py) the model itself
# can say "this is not a fruit" and no threshold is needed for those photos.
REJECT_LABELS = {"not_fruit"}

# Score softening. The classifier is extremely sure of itself, so a plain
# fresh / (fresh + rotten) score is almost always 99-100 or 0-1 - and no real fruit
# is 100% fresh. The two class outputs are softened with a "temperature" so very sure
# results show as roughly 90-98 (or 2-10) instead, and different photos give different
# numbers. 2.0 = default. Set FRESHSENSE_SCORE_SOFTENING=1 to switch it off (original score).
def _softening_from_env():
    try:
        return max(1.0, float(os.environ.get("FRESHSENSE_SCORE_SOFTENING", "2.0")))
    except ValueError:
        return 2.0

SCORE_SOFTENING = _softening_from_env()
SCORE_FLOOR, SCORE_CEILING = 2.0, 98.0   # a score is never shown as exactly 0 or 100 when softening is on


def soften_score(fresh_fraction):
    """Fresh fraction (0..1) -> 0..100 score, softened so that it is never 0 or 100."""
    t = SCORE_SOFTENING
    if t <= 1.0:
        return fresh_fraction * 100.0
    a, b = fresh_fraction ** (1.0 / t), (1.0 - fresh_fraction) ** (1.0 / t)
    score = 100.0 * a / (a + b) if (a + b) > 0 else 50.0
    return min(SCORE_CEILING, max(SCORE_FLOOR, score))


# Light guard. Below this brightness (0 = black, 1 = white; measured in the centre of the photo)
# there is almost no light, so FreshSense refuses to guess. Measured with the real model: a
# banana still gave the same answer at brightness 0.085, but photos around 0.04 or darker
# were answered with a wrong "not a fruit".
MIN_BRIGHTNESS_TO_ANALYSE = 0.07
DARK_PHOTO_BRIGHTNESS = 0.25       # same line the lighting report uses for "too dark"

MAX_FRUITS = 8                     # most fruits reported in one photo
MAX_SIDE = 1280                    # large photos are shrunk to this for speed
_DETECTION_DEBUG = os.environ.get("FRESHSENSE_DEBUG_DETECTION", "1").lower() not in {"0", "false", "off"}


def _detection_log(message):
    """Temporary diagnostics for tracing real multi-fruit segmentation failures."""
    if _DETECTION_DEBUG:
        print(f"[freshsense][detection] {message}", flush=True)

_keras_model = None
_labels = None
_image_size = (224, 224)
_rescale = True                    # True -> divide pixels by 255; False -> model takes raw 0..255
_feat_model = None                 # input -> pooled features
_W = None                          # final dense layer weights (features -> class scores)
_b = None
_detector = None
_detector_checked = False
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IMAGE_SIZE = 640
DETECTOR_FRUITS = {"apple", "banana", "orange"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _prepare_head(model, tf):
    """Expose pooled features + final dense weights so we can read raw class scores (logits)."""
    global _feat_model, _W, _b
    _feat_model = _W = _b = None
    try:
        gaps = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)]
        denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        if not gaps or not denses:
            return
        dense = denses[-1]
        if dense.get_config().get("activation") != "softmax":
            return
        W, b = [w.numpy() for w in dense.weights]
        fm = tf.keras.Model(model.input, gaps[-1].output)
        if fm.output_shape[-1] != W.shape[0]:
            return
        _feat_model, _W, _b = fm, W, b
    except Exception as e:  # not fatal: we fall back to softmax-only gating
        print(f"[freshsense] Could not read model internals, using softmax gate: {e}")
        _feat_model = _W = _b = None


def _labels_from_metadata(metadata):
    """Build output-index labels in the exact order stored with the model."""
    return [f"{item['fruit']}_{item['condition']}" for item in metadata["classes"]]


def _try_load_model():
    """Lazily load a trained model if one exists. Returns None -> fallback mode."""
    global _keras_model, _labels, _image_size, _rescale
    if _keras_model is not None:
        return _keras_model

    try:
        import tensorflow as tf
    except Exception as e:
        print(f"[freshsense] TensorFlow not available, using fallback mode: {e}")
        return None

    if os.path.exists(_KERAS_MODEL_PATH) and os.path.exists(_METADATA_PATH):
        try:
            with open(_METADATA_PATH) as f:
                meta = json.load(f)
            _labels = _labels_from_metadata(meta)
            size = meta.get("image_size", [224, 224])
            _image_size = (int(size[0]), int(size[1]))
            _rescale = "0..255" not in meta.get("input", "") and "0-255" not in meta.get("input", "")
            _keras_model = tf.keras.models.load_model(_KERAS_MODEL_PATH)
            _prepare_head(_keras_model, tf)
            print(f"[freshsense] Loaded trained model: {_KERAS_MODEL_PATH} "
                  f"({len(_labels)} classes, image_size={_image_size}, rescale={_rescale}, "
                  f"logit_gate={'yes' if _feat_model is not None else 'no'})")
            return _keras_model
        except Exception as e:
            print(f"[freshsense] Found {_KERAS_MODEL_PATH} but failed to load it: {e}")
            _keras_model = None
            return None

    if os.path.exists(_LEGACY_MODEL_PATH):
        try:
            _keras_model = tf.keras.models.load_model(_LEGACY_MODEL_PATH)
            if os.path.exists(_LEGACY_LABELS_PATH):
                with open(_LEGACY_LABELS_PATH) as f:
                    _labels = [line.strip() for line in f if line.strip()]
            _image_size = (224, 224)
            _rescale = True
            _prepare_head(_keras_model, tf)
            return _keras_model
        except Exception as e:
            print(f"[freshsense] Could not load trained model: {e}")
            _keras_model = None
            return None
    return None


def supported_fruits():
    """Names of the fruits the loaded model knows (empty if no model)."""
    if _try_load_model() is None or not _labels:
        return []
    seen = []
    for l in _labels:
        if l in REJECT_LABELS:
            continue
        f = l.rsplit("_", 1)[0]
        if f not in seen:
            seen.append(f)
    return seen


def _load_detector():
    """Load the separate COCO YOLO detector lazily; never replace the classifier."""
    global _detector, _detector_checked
    if _detector_checked:
        return _detector
    _detector_checked = True
    if not os.path.isfile(_DETECTOR_PATH):
        _detection_log(f"YOLO detector weights not found at {_DETECTOR_PATH}")
        return None
    try:
        from ultralytics import YOLO
        _detector = YOLO(_DETECTOR_PATH)
        _detection_log(f"Loaded CPU YOLO detector: {_DETECTOR_PATH}")
    except Exception as exc:
        _detector = None
        _detection_log(f"YOLO detector could not load; contour fallback will be used: {exc}")
    return _detector


def _detect_supported_fruits(rgb):
    """Return COCO apple/banana/whole-orange boxes, or None if detector is unavailable."""
    detector = _load_detector()
    if detector is None:
        return None
    try:
        names = detector.names
        if isinstance(names, dict):
            class_ids = [int(idx) for idx, name in names.items()
                         if str(name).lower() in DETECTOR_FRUITS]
        else:
            class_ids = [idx for idx, name in enumerate(names)
                         if str(name).lower() in DETECTOR_FRUITS]
        if not class_ids:
            _detection_log("YOLO weights contain none of the supported COCO fruit classes")
            return []
        # Ask YOLO for low-confidence boxes from all classes, then apply the
        # existing application threshold and supported-class list below. This
        # lets diagnostics show boxes rejected by either application filter,
        # without changing which detections are accepted (0.25 remains the
        # effective confidence threshold).
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        outputs = detector.predict(
            source=bgr, imgsz=DETECTOR_IMAGE_SIZE, conf=0.001,
            device="cpu", verbose=False,
        )
        found = []
        for output in outputs:
            if output.boxes is None:
                continue
            for row in output.boxes:
                class_id = int(row.cls.item())
                food = str(names[class_id] if not isinstance(names, dict)
                           else names[class_id]).lower()
                confidence = float(row.conf.item())
                x1, y1, x2, y2 = [int(round(v)) for v in row.xyxy[0].tolist()]
                raw_bbox = (x1, y1, x2 - x1, y2 - y1)
                if food not in DETECTOR_FRUITS:
                    _detection_log(
                        f"YOLO raw detection rejected: class={food}, confidence={confidence:.4f}, "
                        f"bbox={raw_bbox}, reason=unsupported class"
                    )
                    continue
                if confidence < DETECTOR_CONFIDENCE:
                    _detection_log(
                        f"YOLO raw detection rejected: class={food}, confidence={confidence:.4f}, "
                        f"bbox={raw_bbox}, reason=confidence below {DETECTOR_CONFIDENCE:.2f}"
                    )
                    continue
                x1, x2 = max(0, x1), min(rgb.shape[1], x2)
                y1, y2 = max(0, y1), min(rgb.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    _detection_log(
                        f"YOLO raw detection rejected: class={food}, confidence={confidence:.4f}, "
                        f"bbox={raw_bbox}, reason=invalid box after image-boundary clipping"
                    )
                    continue
                found.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "food": food.capitalize(),
                    "confidence": round(confidence, 4),
                })
        found.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        _detection_log(
            f"YOLO accepted detections={len(found)} (classes=apple,banana,orange; "
            f"conf>={DETECTOR_CONFIDENCE}): "
            f"{[(d['food'], d['confidence'], d['bbox']) for d in found]}"
        )
        return found
    except Exception as exc:
        _detection_log(f"YOLO inference failed; contour fallback will be used: {exc}")
        return None


def _load_gate_config():
    try:
        with open(GATE_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def gate_threshold(kind):
    cfg = _load_gate_config()
    if kind == "max_logit":
        env = os.environ.get("FRESHSENSE_MIN_LOGIT")
        if env is not None:
            return float(env)
        return float(cfg.get("min_max_logit", DEFAULT_MIN_LOGIT))
    return float(cfg.get("min_max_softmax", DEFAULT_MIN_SOFTMAX))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _read_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return arr


def _status_from_score(score):
    if score >= STATUS_THRESHOLDS["fresh_min"]:
        return "Fresh", "🟢"
    if score >= STATUS_THRESHOLDS["moderate_min"]:
        return "Moderate", "🟡"
    return "Spoiled / Not Fresh", "🔴"


def _short_status(status):
    # Preserve the established `overall` API value for existing clients.
    return {"Fresh": "fresh", "Moderate": "moderate"}.get(status, "rotten")


# ---------------------------------------------------------------------------
# Lighting: measure it, warn about it, and (optionally) correct it
# ---------------------------------------------------------------------------
def lighting_report(rgb_image):
    """Measure how good the lighting in the photo is and list any problems."""
    gray = cv2.cvtColor(cv2.resize(rgb_image, (224, 224)), cv2.COLOR_RGB2GRAY).astype("float32")
    # Judge only the CENTRE of the photo, where the fruit usually is. Looking at
    # the whole frame would wrongly flag fruit photographed on a white background.
    centre = gray[45:179, 45:179]
    brightness = float(centre.mean()) / 255.0
    contrast = float(centre.std()) / 255.0
    dark_frac = float((centre < 30).mean())
    clipped_frac = float((centre > 250).mean())     # completely blown-out pixels

    warnings = []
    if brightness < 0.25:
        warnings.append("Photo is too dark - move to better light")
    if clipped_frac > 0.35:
        warnings.append("Photo is overexposed - reduce light or move away from the flash")
    elif clipped_frac > 0.08:
        warnings.append("Glare or blown-out spots on the fruit")
    if contrast < 0.05:
        warnings.append("Flat, low-contrast lighting - spoilage marks may be hidden")
    if dark_frac > 0.40 and brightness >= 0.25:
        warnings.append("Large shadow areas - try even lighting")

    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "quality": "good" if not warnings else "poor",
        "warnings": warnings,
    }


def normalize_lighting(rgb_image):
    """
    Reduce the effect of lighting: gray-world white balance (removes a yellow
    or blue colour cast) + CLAHE on lightness (evens out dark/bright regions).
    Off by default - use lighting_test.py to check whether it helps on your photos.
    """
    img = rgb_image.astype("float32")
    means = img.reshape(-1, 3).mean(axis=0)
    gray_mean = means.mean()
    img = np.clip(img * (gray_mean / np.maximum(means, 1e-6)), 0, 255).astype("uint8")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# OpenCV colour analysis (the fallback when there is no trained model)
# ---------------------------------------------------------------------------
def _heuristic_freshness(rgb_image):
    """Returns (freshness_score 0-100, status, emoji, details). Pure OpenCV/NumPy."""
    img = cv2.resize(rgb_image, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    dark_ratio = float(np.mean(v < 70))                                        # bruising / rot / mould
    brown_ratio = float(np.mean((h > 5) & (h < 30) & (v < 120) & (s > 40)))    # brown blemishes
    avg_sat = float(np.mean(s)) / 255.0                                        # fresh = vivid
    avg_val = float(np.mean(v)) / 255.0

    score = 100.0 - (dark_ratio * 50) - (brown_ratio * 60) - ((1 - avg_sat) * 20) - ((1 - avg_val) * 15)
    score = max(0.0, min(100.0, score))
    status, emoji = _status_from_score(score)

    details = {
        "dark_spot_ratio": round(dark_ratio, 3),
        "brown_blemish_ratio": round(brown_ratio, 3),
        "avg_saturation": round(avg_sat, 3),
        "avg_brightness": round(avg_val, 3),
    }
    return round(score, 1), status, emoji, details


# ---------------------------------------------------------------------------
# Finding the separate fruits in a photo
# ---------------------------------------------------------------------------
def _merge_nested(boxes):
    """boxes: [x0, y0, x1, y1, area]. Merge a box into another if its centre lies inside it."""
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                a, b = boxes[i], boxes[j]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if a[0] <= cx <= a[2] and a[1] <= cy <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]),
                                max(a[3], b[3]), a[4] + b[4]]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def _split_large_region(lab_image, foreground_mask, box, min_area, open_kernel):
    """Try splitting a large outer contour, such as a plate, into objects inside it.

    This only accepts a split when the outer region has a clear dominant color
    and at least two sizeable inner components. Otherwise callers keep the
    original region. It is a conservative contour refinement, not a detector.
    """
    x0, y0, x1, y1, _ = box
    roi_lab = lab_image[y0:y1, x0:x1]
    roi_mask = foreground_mask[y0:y1, x0:x1]
    pixels = roi_lab[roi_mask > 0].astype("float32")
    if len(pixels) < 100:
        _detection_log(f"large box {box[:4]} not split: only {len(pixels)} foreground pixels")
        return None

    # Sample deterministically so a large plate doesn't require clustering every pixel.
    if len(pixels) > 5000:
        pixels = pixels[np.random.RandomState(0).choice(len(pixels), 5000, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=len(centers))
    dominant_index = int(np.argmax(counts))
    dominant_fraction = counts[dominant_index] / len(labels)
    if dominant_fraction < 0.40:
        _detection_log(
            f"large box {box[:4]} not split: dominant-color fraction {dominant_fraction:.3f} < 0.400"
        )
        return None

    distances = np.linalg.norm(roi_lab - centers[dominant_index], axis=2)
    roi_distances = distances[roi_mask > 0]
    scaled = np.clip(roi_distances * 255 / 100, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu * 100 / 255, 15, 40))
    inner_mask = ((distances > threshold) & (roi_mask > 0)).astype(np.uint8) * 255
    inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    children = []
    _detection_log(
        f"large box {box[:4]} inner raw contours={len(contours)}, area cutoff={min_area:.1f}px"
    )
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            _detection_log(f"  inner contour {index + 1} rejected: area {area:.1f}px < {min_area:.1f}px")
            continue
        x, y, width, height = cv2.boundingRect(contour)
        child = [x0 + x, y0 + y, x0 + x + width, y0 + y + height, area]
        children.append(child)
        _detection_log(f"  inner contour {index + 1} accepted: bbox={child[:4]}, area={area:.1f}px")

    if len(children) < 2:
        _detection_log(f"large box {box[:4]} not split: only {len(children)} inner candidates survived")
        return None
    return children


def find_fruit_regions(rgb):
    """
    Split a photo into separate objects (candidate fruits).

    Works when the objects sit on a fairly plain background (a table, a white
    sheet, a plate). Returns ([(x, y, w, h), ...], mode). On a busy background
    it returns ([], "complex-background") and the whole photo is analysed as one.
    Fruits that touch each other are treated as ONE object.
    """
    h, w = rgb.shape[:2]
    scale = 480.0 / max(h, w) if max(h, w) > 480 else 1.0
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sh, sw = small.shape[:2]
    _detection_log(f"find_fruit_regions input={w}x{h}, working={sw}x{sh}, scale={scale:.4f}")
    lab = cv2.cvtColor(small.astype("float32") / 255.0, cv2.COLOR_RGB2LAB)

    m = max(2, int(0.03 * min(sh, sw)))
    border = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                             lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    # Background colour = the biggest colour group along the photo's edge. (Fruit may
    # touch or cross the edge, so a plain median of the border is not safe.)
    pts = border.astype("float32")
    if len(pts) > 4000:
        pts = pts[np.random.RandomState(0).choice(len(pts), 4000, replace=False)]
    _, labels, centers = cv2.kmeans(pts, 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    big = int(np.argmax(np.bincount(labels, minlength=2)))
    group = pts[labels == big]
    if len(group) < 0.5 * len(pts):
        _detection_log(f"background rejected: dominant edge cluster={len(group)}/{len(pts)} (<50%)")
        return [], "complex-background"
    bg = np.median(group, axis=0)
    edge_spread = float(np.percentile(np.linalg.norm(group - bg, axis=1), 85))
    if edge_spread > 15:
        _detection_log(f"background rejected: 85th-percentile edge color spread={edge_spread:.2f} (>15)")
        return [], "complex-background"

    dist = np.linalg.norm(lab - bg, axis=2)                 # colour distance from background
    d8 = np.clip(dist * 255 / 100, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu * 100 / 255, 15, 40))
    mask = (dist > thr).astype(np.uint8) * 255

    k = max(3, int(0.012 * min(sh, sw))) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.025 * sh * sw
    _detection_log(f"outer contours before area filtering={len(contours)}, area cutoff={min_area:.1f}px")
    boxes = []
    for index, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        if area < min_area:
            _detection_log(
                f"  outer contour {index + 1} rejected: bbox={[x, y, bw, bh]}, area={area:.1f}px < {min_area:.1f}px"
            )
            continue
        boxes.append([x, y, x + bw, y + bh, area])
        _detection_log(f"  outer contour {index + 1} accepted: bbox={[x, y, bw, bh]}, area={area:.1f}px")
    before_merge = len(boxes)
    boxes = _merge_nested(boxes)
    _detection_log(f"candidate boxes after area filtering={before_merge}; after nested-box merge={len(boxes)}")

    # A plate can be the largest foreground contour and enclose multiple fruits.
    # For unusually large boxes, try using the dominant interior surface color
    # as a local background and retain only clearly separate inner components.
    frame_area = sh * sw
    for index, box in enumerate(boxes):
        box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        # The mixed-fruit regression photo's joined fruit contour occupies 39.5%
        # of the working frame, just below the previous 40% trigger. Keep the
        # inner-component checks in _split_large_region before replacing boxes.
        if box_area < 0.39 * frame_area:
            continue
        _detection_log(f"attempting inner split for large candidate bbox={box[:4]}, box_fraction={box_area / frame_area:.3f}")
        split = _split_large_region(lab, mask, box, min_area, np.ones((k, k), np.uint8))
        if split:
            _detection_log(f"large candidate bbox={box[:4]} replaced by {len(split)} inner boxes")
            boxes[index:index + 1] = split

    if not boxes:
        _detection_log("no candidates remained after contour-area filtering and refinement")
        return [], "no-objects-found"
    if len(boxes) == 1 and boxes[0][4] > 0.92 * sh * sw:
        _detection_log(f"sole candidate rejected as full-frame region: bbox={boxes[0][:4]}, area={boxes[0][4]:.1f}px")
        return [], "complex-background"          # "object" is the whole frame -> segmentation failed

    boxes.sort(key=lambda b: -b[4])
    _detection_log(f"candidate regions before MAX_FRUITS cap={len(boxes)}; cap={MAX_FRUITS}")
    for index, box in enumerate(boxes[MAX_FRUITS:], start=MAX_FRUITS + 1):
        _detection_log(f"  candidate {index} rejected by MAX_FRUITS: bbox={box[:4]}, area={box[4]:.1f}px")
    boxes = boxes[:MAX_FRUITS]
    out = [(int(b[0] / scale), int(b[1] / scale), int((b[2] - b[0]) / scale), int((b[3] - b[1]) / scale))
           for b in boxes]
    row = max(1, int(0.2 * h))                    # reading order: top-to-bottom rows, then left-to-right
    out.sort(key=lambda b: ((b[1] + b[3] // 2) // row, b[0]))
    _detection_log(f"candidate regions returned={len(out)}; boxes={out}")
    return out, "plain-background"


DEFAULT_CROP_PADDING = 0.03
MAX_CROP_PADDING = 0.10


def _crop_padding_fraction():
    """Read a small per-side crop margin, capped to prevent accidental wide crops."""
    try:
        value = float(os.environ.get("FRESHSENSE_CROP_PADDING", DEFAULT_CROP_PADDING))
    except (TypeError, ValueError):
        value = DEFAULT_CROP_PADDING
    return min(MAX_CROP_PADDING, max(0.0, value))


def _crop_region(rgb, box, pad_frac=None, nearby_boxes=None):
    """Crop a detector box with small padding, clipping padding before nearby boxes.

    The returned crop stays rectangular. `_run_model` resizes it to the model's
    square input dimensions using training-compatible bilinear interpolation,
    so making a square here would needlessly pull in scene pixels.
    """
    h, w = rgb.shape[:2]
    x, y, bw, bh = [int(round(value)) for value in box]
    bx0, by0 = max(0, x), max(0, y)
    bx1, by1 = min(w, x + bw), min(h, y + bh)
    if bx1 <= bx0 or by1 <= by0:
        raise ValueError(f"Crop box {box!r} does not intersect image bounds {w}x{h}")

    pad_frac = _crop_padding_fraction() if pad_frac is None else min(MAX_CROP_PADDING, max(0.0, float(pad_frac)))
    pad_x = int(round((bx1 - bx0) * pad_frac))
    pad_y = int(round((by1 - by0) * pad_frac))
    x0, y0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    x1, y1 = min(w, bx1 + pad_x), min(h, by1 + pad_y)

    # Retain the detector box itself; only trim optional padding when that
    # padding would cross the edge of another detected object.
    for other in nearby_boxes or []:
        ox, oy, ow, oh = [int(round(value)) for value in other]
        if (max(0, ox), max(0, oy), min(w, ox + ow), min(h, oy + oh)) == (bx0, by0, bx1, by1):
            continue
        ox0, oy0 = max(0, ox), max(0, oy)
        ox1, oy1 = min(w, ox + ow), min(h, oy + oh)
        if ox1 <= ox0 or oy1 <= oy0:
            continue

        vertical_overlap = y0 < oy1 and y1 > oy0
        horizontal_overlap = x0 < ox1 and x1 > ox0
        if vertical_overlap and ox1 <= bx0 and ox1 > x0:
            x0 = max(x0, ox1)
        if vertical_overlap and ox0 >= bx1 and ox0 < x1:
            x1 = min(x1, ox0)
        if horizontal_overlap and oy1 <= by0 and oy1 > y0:
            y0 = max(y0, oy1)
        if horizontal_overlap and oy0 >= by1 and oy0 < y1:
            y1 = min(y1, oy0)

    crop = rgb[y0:y1, x0:x1].copy()
    _detection_log(
        f"crop bbox={tuple(box)}, crop_rect={(x0, y0, x1 - x0, y1 - y0)}, "
        f"crop_pixels={crop.shape[1]}x{crop.shape[0]}, pad_fraction={pad_frac:.3f}"
    )
    return crop


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------
def _run_model(images):
    """
    Run the model on a list of RGB images (one batch).
    Returns (probs [N x classes], gate_scores [N], gate_kind).
    """
    model = _try_load_model()
    # image_dataset_from_directory defaults to bilinear resizing during training.
    # Keep the same interpolation at inference; inputs here are already RGB.
    batch = np.stack([cv2.resize(i, _image_size, interpolation=cv2.INTER_LINEAR) for i in images]).astype("float32")
    if _rescale:
        batch = batch / 255.0
    if _feat_model is not None:
        feats = _feat_model.predict(batch, verbose=0)
        z = feats @ _W + _b
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True), z.max(axis=1), "max_logit"
    probs = model.predict(batch, verbose=0)
    return probs, probs.max(axis=1), "max_softmax"


def _interpret(prob_row, expected_food=None):
    """Turn classifier outputs into a fruit name, score, and uncalibrated model scores."""
    fruit_prob, cond_prob, class_scores = {}, {}, {}
    for i, p in enumerate(prob_row):
        label = _labels[i] if _labels else f"class_{i}"
        if label in REJECT_LABELS:
            continue
        fruit, cond = label.rsplit("_", 1) if "_" in label else (label, "unknown")
        fruit_prob[fruit] = fruit_prob.get(fruit, 0.0) + float(p)
        cond_prob.setdefault(fruit, {})[cond.lower()] = float(p)
        class_scores.setdefault(fruit, {})[cond.lower()] = float(p)

    top_food = max(fruit_prob, key=fruit_prob.get)
    expected_key = str(expected_food).lower() if expected_food else None
    food = expected_key if expected_key in fruit_prob else top_food
    conds = cond_prob[food]
    fresh_p = conds.get("fresh", 0.0)
    bad_p = conds.get("rotten", 0.0) + conds.get("spoiled", 0.0)
    moderate_p = conds.get("moderate", 0.0)
    total = fresh_p + bad_p + moderate_p
    numerator = fresh_p + 0.5 * moderate_p
    score_before_softening = round((numerator / total) * 100, 1) if total > 0 else 0.0
    score = round(soften_score(numerator / total), 1) if total > 0 else 0.0
    all_class_scores = {
        (_labels[i] if _labels and i < len(_labels) else f"class_{i}"): float(p)
        for i, p in enumerate(prob_row)
    }
    return {
        "food": food.capitalize(),
        "classifier_top_food": top_food.capitalize(),
        "score": score,
        "score_before_softening": score_before_softening,
        "fruit_model_score": fruit_prob[food],
        "fresh_model_score": fresh_p / total if total > 0 else 0.0,
        "rotten_model_score": bad_p / total if total > 0 else 0.0,
        "score_numerator": numerator,
        "score_denominator": total,
        "all_class_scores": all_class_scores,
        # Keep the actual selected-fruit output columns visible for diagnosis.
        # These are classifier outputs, not calibrated real-world probabilities.
        "raw_class_scores": {
            f"{food}_{condition}": class_scores[food][condition]
            for condition in ("fresh", "rotten") if condition in class_scores[food]
        },
    }


def _fruit_item(idx, box, interp, gate_score, weak=False, detection=None):
    status, emoji = _status_from_score(interp["score"])
    item = {
        "id": idx,
        "bbox": [int(v) for v in box],
        "food": interp["food"],
        "freshness_score": interp["score"],
        "status": status,
        "status_icon": emoji,
        "fruit_model_score": round(interp["fruit_model_score"], 3),
        "score_basis": "model_output_score",
        "status_basis": STATUS_BASIS,
        "gate_score": round(float(gate_score), 2),
        "details": {
            "fruit_model_score": round(interp["fruit_model_score"], 3),
            "fresh_model_score": round(interp["fresh_model_score"], 3),
            "rotten_model_score": round(interp["rotten_model_score"], 3),
            "raw_class_scores": {k: round(v, 6) for k, v in interp["raw_class_scores"].items()},
            "model_score_note": "Raw softmax model outputs; not calibrated real-world probabilities.",
            "score_before_softening": interp.get("score_before_softening"),
        },
    }
    if detection is not None:
        item["detector_confidence"] = detection["confidence"]
        item["food_source"] = "yolo11n_coco_detector"
        if detection["food"].lower() == "orange":
            item["detection_note"] = (
                "The COCO orange label does not distinguish whole oranges from slices; "
                "slice detection and freshness results are unverified."
            )
        if interp["classifier_top_food"].lower() != detection["food"].lower():
            item["classifier_food_prediction"] = interp["classifier_top_food"]
            item["identification_warning"] = (
                f"Fruit identification uncertain: the detector labeled this {detection['food']}, "
                f"but the classifier's top fruit label was {interp['classifier_top_food']}."
            )
            item["classifier_note"] = (
                f"The object detector labeled this {detection['food']}; the freshness score uses "
                f"that fruit's classifier outputs. The classifier's top fruit label was "
                f"{interp['classifier_top_food']}."
            )
    if interp["fruit_model_score"] < 0.6:
        item["note"] = "Low fruit model score - try a clearer, closer photo of this fruit"
    elif weak:
        item["note"] = "The fruit model score is near the rejection threshold - check the result"
    if "classifier_note" in item:
        item["note"] = f"{item['note']} {item['classifier_note']}" if "note" in item else item["classifier_note"]
    # "How soon will it spoil?" - rough estimate from the fruit type and its freshness score.
    uncertain = "identification_warning" in item or interp["fruit_model_score"] < 0.6 or weak
    estimate = shelf_life.estimate(interp["food"], interp["score"], uncertain=uncertain)
    if estimate:
        item["shelf_life"] = estimate
    return item


# ---------------------------------------------------------------------------
# Annotated picture (boxes + labels drawn on the photo)
# ---------------------------------------------------------------------------
_STATUS_COLOR = {"Fresh": (46, 160, 67), "Moderate": (230, 160, 0)}
_ROTTEN_COLOR = (200, 40, 40)
_REJECT_COLOR = (130, 130, 130)


def _annotation_label_rectangles(image_width, image_height, entries):
    """Return measured label rectangles, keeping each in bounds and non-overlapping.

    Entries are ``(bbox, (full_width, full_height), (compact_width, compact_height))``.
    Labels first try above their box, then use a compact size inside the box.
    """
    placed = []
    gap = 3

    for box, full_size, compact_size in entries:
        x, y, bw, bh = [int(v) for v in box]
        full_w, full_h = (min(int(full_size[0]), image_width), min(int(full_size[1]), image_height))
        compact_w, compact_h = (min(int(compact_size[0]), image_width), min(int(compact_size[1]), image_height))

        def available(left, top, label_width, label_height):
            max_x, max_y = image_width - label_width, image_height - label_height
            if left < 0 or top < 0 or left > max_x or top > max_y:
                return False
            right, bottom = left + label_width, top + label_height
            return all(right <= ox or ox + ow <= left or bottom <= oy or oy + oh <= top
                       for ox, oy, ow, oh, *_ in placed)

        # First choice: directly above the box, clamped to the image edges.
        above_y = y - full_h - gap
        above_xs = (x, x + bw - full_w, x + (bw - full_w) // 2)
        above_xs = tuple(min(max(0, left), image_width - full_w) for left in above_xs)
        position = next(((left, above_y, full_w, full_h, False)
                         for left in above_xs
                         if available(left, above_y, full_w, full_h)), None)

        # If other labels occupy the above-box slot, try a smaller label fully
        # inside this fruit's box before looking elsewhere.
        if position is None and bw >= compact_w and bh >= compact_h:
            inside_xs = (x, x + bw - compact_w, x + (bw - compact_w) // 2)
            inside_ys = (y + gap, y + bh - compact_h - gap)
            position = next(((left, top, compact_w, compact_h, True)
                             for top in inside_ys for left in inside_xs
                             if available(left, top, compact_w, compact_h)), None)

        # Try the same above-box row at another free x position.
        if position is None and 0 <= above_y <= image_height - full_h:
            max_x = image_width - full_w
            position = next(((left, above_y, full_w, full_h, False)
                             for left in range(0, max_x + 1, 2)
                             if available(left, above_y, full_w, full_h)), None)

        # Final fallback: nearest free in-image slot, using the compact size.
        if position is None:
            max_x, max_y = image_width - compact_w, image_height - compact_h
            step_y = max(1, compact_h // 2)
            step_x = max(1, min(compact_w // 2, 16))
            position = next(((left, top, compact_w, compact_h, True)
                             for top in range(0, max_y + 1, step_y)
                             for left in range(0, max_x + 1, step_x)
                             if available(left, top, compact_w, compact_h)), None)
        if position is not None:
            placed.append(position)

    return placed


def _annotate(rgb, fruits, rejected):
    h, w = rgb.shape[:2]
    s = 720.0 / max(h, w) if max(h, w) > 720 else 1.0
    img = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb.copy()
    th = max(2, int(round(min(img.shape[:2]) / 160)))
    fs = max(0.5, min(img.shape[:2]) / 700)

    def draw_box(box, color):
        x, y, bw, bh = [int(v * s) for v in box]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, th)
        return [int(v * s) for v in box]

    labels = []
    for fruit in fruits:
        color = _STATUS_COLOR.get(fruit["status"], _ROTTEN_COLOR)
        box = draw_box(fruit["bbox"], color)
        labels.append((box, str(fruit["id"]), color))
    for box in rejected:
        scaled_box = draw_box(box, _REJECT_COLOR)
        labels.append((scaled_box, "!", _REJECT_COLOR))

    # The image is shown at a much smaller CSS width in the browser. Use compact
    # numbered tags here; the Fruit 1 / Fruit 2 cards below provide the full legend.
    tag_size = min(max(30, int(round(32 * s))), img.shape[1], img.shape[0])
    label_entries = [(box, (tag_size, tag_size), (tag_size, tag_size))
                     for box, _, _ in labels]
    rectangles = _annotation_label_rectangles(img.shape[1], img.shape[0], label_entries)
    for (box, text, color), rect in zip(labels, rectangles):
        left, top, tag_w, tag_h, _compact = rect
        cv2.rectangle(img, (left, top), (left + tag_w - 1, top + tag_h - 1), color, -1)
        font_scale = max(0.5, min(tag_w, tag_h) * 0.045)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = left + (tag_w - text_w) // 2
        text_y = top + (tag_h + text_h) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Building the final answer
# ---------------------------------------------------------------------------
def _summarise(fruits):
    """Summarize the score bands without presenting them as verified conditions."""
    n = len(fruits)
    if n == 0:
        return "none", "No fruit found"
    overall_words = [_short_status(f["status"]) for f in fruits]
    if n == 1:
        f = fruits[0]
        return overall_words[0], (f"Visual estimate: {f['food']} falls in the {f['status']} score band "
                          f"({f['freshness_score']}/100).")
    display_words = ["spoiled / not fresh" if f["status"] == "Spoiled / Not Fresh"
                     else f["status"].lower() for f in fruits]
    counts = {}
    for wd in display_words:
        counts[wd] = counts.get(wd, 0) + 1
    breakdown = ", ".join(f"{c} {wd}" for wd, c in counts.items())
    each = "; ".join(f"Fruit {f['id']} ({f['food']}) falls in the {f['status']} score band"
                      for f in fruits)
    if len(counts) == 1:
        return overall_words[0], f"Visual estimate: {n} fruits fall in the same {display_words[0]} score band. {each}."
    return "mixed", f"Visual estimate: {n} fruits fall in mixed score bands ({breakdown}). {each}."


def _not_fruit_message():
    names = ", ".join(supported_fruits()) or "apple, banana, grape, guava, jujube, orange, pomegranate, strawberry"
    return ("This does not look like a fruit FreshSense knows. It may not be a fruit at all, "
            f"or it is a fruit outside the supported list ({names}).")


def _build_result(rgb, fruits, rejected, seg_mode, lighting, normalize, closest=None, best_gate=None,
                  detection_source="opencv_contour_fallback", fallback_reason=None):
    n = len(fruits)
    overall, text = _summarise(fruits)
    result = {
        "mode": "trained_model",
        "is_fruit": n > 0,
        "fruit_count": n,
        "overall": overall,
        "summary": text,
        "message": text if n else _not_fruit_message(),
        "fruits": fruits,
        "segmentation": seg_mode,
        "detection_source": detection_source,
        "detection_note": (
            "Pretrained YOLO11n COCO detects apple, banana, and orange classes. "
            "Grapes and orange slices are not supported or verified."
            if detection_source == "yolo11n_coco_detector" else None
        ),
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "fallback_reason": fallback_reason,
        "score_basis": "model_output_score",
        "status_basis": STATUS_BASIS,
        "status_thresholds": STATUS_THRESHOLDS,
        "disclaimer": RESULT_DISCLAIMER,
        "lighting": lighting,
        "lighting_normalized": bool(normalize),
        "annotated_image": _annotate(rgb, fruits, rejected),
    }
    # Top-level fields kept so simple clients keep working.
    if n == 0:
        result.update(food="Not a fruit", freshness_score=None, status="Not a fruit",
                      status_icon="⚪", fruit_model_score=None,
                      details={"closest_class_not_trusted": closest, "best_gate_score": best_gate})
    elif n == 1:
        f = fruits[0]
        result.update(food=f["food"], freshness_score=f["freshness_score"], status=f["status"],
                      status_icon=f["status_icon"], fruit_model_score=f["fruit_model_score"], details=f["details"], shelf_life=f.get("shelf_life"))
    else:
        worst = min(fruits, key=lambda f: f["freshness_score"])
        if overall == "mixed":
            status, icon = "Mixed", "🟠"
        else:
            status, icon = worst["status"], worst["status_icon"]
        # freshness_score = the lowest score band among the fruits.
        result.update(food=f"{n} fruits", freshness_score=worst["freshness_score"], status=status,
                      status_icon=icon, fruit_model_score=None,
                      details={"lowest_score_among_fruits": worst["freshness_score"]})
    return result


def _too_dark_result(rgb, lighting, normalize):
    """Almost no light: say so instead of guessing (or wrongly saying 'not a fruit')."""
    msg = ("This photo is too dark to analyse - there is almost no light. "
           "Please retake it in better light.")
    return {
        "mode": "not_analysed", "analysis_blocked": "too_dark", "is_fruit": None, "fruit_count": 0,
        "overall": "none", "summary": msg, "message": msg, "fruits": [],
        "segmentation": "not-run", "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "detection_source": "not_run_too_dark", "fallback_reason": None,
        "score_basis": None, "status_basis": STATUS_BASIS, "status_thresholds": STATUS_THRESHOLDS,
        "disclaimer": RESULT_DISCLAIMER, "lighting": lighting, "lighting_normalized": bool(normalize),
        "annotated_image": None,
        "food": "Unknown", "freshness_score": None, "status": "Too dark", "status_icon": "🌑",
        "fruit_model_score": None, "details": {"brightness": lighting["brightness"]},
    }


def _fallback_result(rgb, lighting, normalize, reason):
    """No usable trained model: colour-based estimate only. We cannot confirm it is a fruit."""
    score, status, emoji, details = _heuristic_freshness(rgb)
    fruit = {"id": 1, "bbox": [0, 0, int(rgb.shape[1]), int(rgb.shape[0])], "food": "Unknown",
             "freshness_score": score, "status": status, "status_icon": emoji,
             "fruit_model_score": None, "score_basis": "heuristic_visual_features",
             "status_basis": STATUS_BASIS, "gate_score": None, "details": details}
    msg = (f"Heuristic visual estimate ({reason}). Colour features only; FreshSense cannot identify "
           "the fruit in this mode. The score bands are experimental and do not establish food safety.")
    fruit["score_basis"] = "heuristic_visual_features"
    fruit["status_basis"] = STATUS_BASIS
    return {
        "mode": "fallback_heuristic", "is_fruit": None, "fruit_count": 1,
        "overall": _short_status(status), "summary": msg, "message": msg, "fruits": [fruit],
        "segmentation": "whole-image", "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "detection_source": "heuristic_whole_image_fallback",
        "fallback_reason": reason, "lighting": lighting, "lighting_normalized": bool(normalize),
        "score_basis": "heuristic_visual_features", "status_basis": STATUS_BASIS,
        "status_thresholds": STATUS_THRESHOLDS, "disclaimer": RESULT_DISCLAIMER,
        "annotated_image": _annotate(rgb, [fruit], []),
        "food": "Unknown", "freshness_score": score, "status": status, "status_icon": emoji,
        "fruit_model_score": None, "details": details,
    }


def looks_like_graphic(rgb):
    """
    True for charts, text pages, logos, plain textures and other flat-colour images.
    A real photo has thousands of slightly different shades (shading, gradients, camera
    noise); a chart or document has only a few flat colours. Measured: real fruit photos
    use ~0.6-0.8 distinct colours per pixel (after shrinking), charts/text/logos < 0.06.
    """
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (160, max(2, int(160 * h / max(w, 1)))), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype("int32")
    distinct = len(np.unique(px[:, 0] * 65536 + px[:, 1] * 256 + px[:, 2]))
    return bool(distinct / len(px) < 0.12)


def _analyze_detected_regions(rgb, detections, lighting, normalize):
    """Use detector boxes for object identity, then the existing model for condition."""
    # Keep the existing response's eight-result ceiling.
    detections = sorted(
        detections,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )[:MAX_FRUITS]
    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    boxes = [tuple(item["bbox"]) for item in detections]
    crops = [_crop_region(rgb, box, nearby_boxes=boxes) for box in boxes]
    _detection_log(
        f"YOLO crops sent to freshness classifier={len(crops)}: "
        f"{[(d['food'], d['confidence'], d['bbox']) for d in detections]}"
    )

    probs, gate, kind = _run_model(crops)
    fruits = []
    for index, (detection, box, row) in enumerate(zip(detections, boxes, probs), start=1):
        interp = _interpret(row, expected_food=detection["food"])
        _detection_log(
            f"  YOLO crop {index}: bbox={box}, detector_food={detection['food']}, "
            f"detector_confidence={detection['confidence']:.4f}, "
            f"selected_fruit_class={interp['food']}, "
            f"classifier_top_food={interp['classifier_top_food']}, "
            f"selected_fruit_condition_scores={interp['raw_class_scores']}, "
            f"score_calculation=(fresh+0.5*moderate)/total*100="
            f"({interp['score_numerator']:.8f}/{interp['score_denominator']:.8f})*100="
            f"{interp['score']}, complete_raw_classifier_scores={interp['all_class_scores']}"
        )
        item = _fruit_item(
            index, box, interp, gate[index - 1],
            weak=(kind == "max_logit" and gate[index - 1] < gate_threshold(kind) + WEAK_GATE_MARGIN),
            detection=detection,
        )
        fruits.append(item)

    return _build_result(
        rgb, fruits, [], "yolo11n-coco", lighting, normalize,
        best_gate=round(float(gate.max()), 2),
        detection_source="yolo11n_coco_detector",
    )


def _analyze_with_model(rgb, lighting, normalize, original=None):
    h, w = rgb.shape[:2]
    if looks_like_graphic(original if original is not None else rgb):                      # chart / text / logo / flat texture: not a fruit photo
        res = _build_result(
            rgb, [], [(0, 0, w, h)], "flat-graphic", lighting, normalize,
            detection_source="not_run_flat_graphic",
        )
        res["message"] = ("This looks like a chart, document or flat graphic, not a photo of a fruit. "
                          "Please upload a real photo of a fruit.")
        return res
    detections = _detect_supported_fruits(rgb)
    if detections:
        return _analyze_detected_regions(rgb, detections, lighting, normalize)

    if detections is None:
        detection_fallback_reason = "YOLO detector unavailable or failed; used contour segmentation"
    else:
        detection_fallback_reason = "YOLO found no supported boxes; used contour segmentation"
    _detection_log(detection_fallback_reason)
    regions, seg_mode = find_fruit_regions(rgb)
    _detection_log(f"analysis received {len(regions)} candidate regions (segmentation={seg_mode})")
    for region_number, region in enumerate(regions, start=1):
        _detection_log(f"  classifier crop {region_number}: bbox={region}")
    boxes = [(0, 0, w, h)] + regions
    crops = [rgb] + [_crop_region(rgb, box, nearby_boxes=regions) for box in regions]

    probs, gate, kind = _run_model(crops)          # one batch: whole photo + every object
    thr = gate_threshold(kind)
    _detection_log(f"classifier batch crops={len(crops)} (includes whole-photo crop), gate={kind}, threshold={thr}")
    interps = [_interpret(p) for p in probs]

    reject_idx = [i for i, l in enumerate(_labels or []) if l in REJECT_LABELS]
    reject_prob = probs[:, reject_idx].sum(axis=1) if reject_idx else np.zeros(len(probs))

    for crop_index, (box, interp) in enumerate(zip(boxes, interps)):
        _detection_log(
            f"  crop {crop_index} model output: bbox={box}, fruit={interp['food']}, "
            f"freshness_score={interp['score']}, raw_class_scores={interp['raw_class_scores']}, "
            f"gate_score={float(gate[crop_index]):.4f}, "
            f"not_fruit_score={float(reject_prob[crop_index]):.4f}"
        )

    def is_fruit(i):
        """Gate: strong enough class score AND the model does not call it 'not_fruit'."""
        return bool(gate[i] >= thr and reject_prob[i] <= 0.5)

    def item(idx, i):
        weak = kind == "max_logit" and gate[i] < thr + WEAK_GATE_MARGIN
        return _fruit_item(idx, boxes[i], interps[i], gate[i], weak=weak)

    fruits, rejected = [], []
    if regions:
        for i in range(1, len(boxes)):
            if is_fruit(i):
                _detection_log(
                    f"  crop {i} accepted: bbox={boxes[i]}, fruit={interps[i]['food']}, "
                    f"score={interps[i]['score']}, gate_score={float(gate[i]):.4f}"
                )
                fruits.append(item(len(fruits) + 1, i))
            else:
                reasons = []
                if gate[i] < thr:
                    reasons.append(f"gate score {float(gate[i]):.4f} < {thr}")
                if reject_prob[i] > 0.5:
                    reasons.append(f"not_fruit output {float(reject_prob[i]):.4f} > 0.5")
                _detection_log(
                    f"  crop {i} rejected: bbox={boxes[i]}, " + ("; ".join(reasons) or "unknown gate reason")
                )
                rejected.append(boxes[i])
        if not fruits and is_fruit(0):              # objects rejected but the whole photo is a fruit
            _detection_log("all region crops rejected; whole-photo crop accepted as one result")
            fruits = [item(1, 0)]
            rejected = []
            seg_mode += " (used whole photo)"
    else:
        if is_fruit(0):
            _detection_log(f"no regions returned; whole-photo crop accepted as {interps[0]['food']}")
            fruits = [item(1, 0)]
        else:
            _detection_log(f"no regions returned; whole-photo crop rejected with gate score {float(gate[0]):.4f}")
            rejected = [boxes[0]]

    closest = None
    if not fruits:
        i = int(np.argmax(probs[0]))
        closest = (_labels[i] if _labels else None)
    return _build_result(rgb, fruits, rejected, seg_mode, lighting, normalize,
                         closest=closest, best_gate=round(float(gate.max()), 2),
                         detection_source="opencv_contour_fallback",
                         fallback_reason=detection_fallback_reason)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _attach_grids(rgb, result):
    """Add the rot grid (which squares look rotten) to every fruit in the result."""
    for f in result.get("fruits", []):
        try:
            x, y, bw, bh = f["bbox"]
            crop = rgb[max(0, y):y + bh, max(0, x):x + bw]
            mode = "absolute" if f["status"].startswith("Spoiled") else "relative"      # only a spoiled fruit may be marked all over
            g = grid_map.build_grid(crop, f["food"], _run_model, _interpret, mode=mode)
            if g:
                pct = g["rotten_percent"]
                if (f["status"].startswith("Spoiled") and pct < 25) or (f["status"] == "Fresh" and pct > 50):
                    g["disagrees_with_whole_fruit"] = True
                    g["summary"] += " The squares disagree with the whole-fruit result, so treat the grid with caution."
                f["rot_grid"] = g
        except Exception as e:                     # the grid is an extra: never break the main result
            _detection_log(f"rot grid skipped: {e}")


def analyze_image(file_bytes, normalize=False, with_grid=True):
    """
    Analyse one photo. Returns a JSON-ready dict:

      is_fruit     True / False (None in fallback mode)
      fruit_count  how many fruits were found
      fruits       one entry per fruit: food, freshness_score, status, bbox ...
      overall      "fresh" / "moderate" / "spoiled" / "mixed" / "none"
      message      one plain-English sentence for the user
      annotated_image  the photo with boxes and labels (data URI)
      lighting     lighting report

    A photo whose fruits fall into different score bands is reported as MIXED,
    with a separate result for each fruit.
    """
    rgb = _read_image(file_bytes)
    lighting = lighting_report(rgb)
    brightness = lighting.get("brightness")
    if brightness is not None and brightness < MIN_BRIGHTNESS_TO_ANALYSE:      # (no light at all: do not guess)
        _detection_log(f"photo too dark (brightness {brightness}); analysis not run")
        return _too_dark_result(rgb, lighting, normalize)
    rgb_in = normalize_lighting(rgb) if normalize else rgb

    if _try_load_model() is None:
        result = _fallback_result(rgb_in, lighting, normalize, "No trained model found")
        _detection_log(
            f"API analysis result: mode={result['mode']}, fruit_count={result['fruit_count']}, "
            f"fruits={[(f.get('id'), f.get('food'), f.get('bbox')) for f in result['fruits']]}"
        )
        return result
    try:
        result = _analyze_with_model(rgb_in, lighting, normalize, original=rgb)
        if with_grid and result.get("fruits"):
            _attach_grids(rgb_in, result)
        if (result.get("is_fruit") is False and brightness is not None and brightness < DARK_PHOTO_BRIGHTNESS
                and result.get("detection_source") != "not_run_flat_graphic"):
            result["dark_photo"] = True
            result["message"] = ("No fruit was found, but this photo is very dark, so a fruit may simply "
                                 "not be visible. Please retake it in better light.")
        _detection_log(
            f"API analysis result: mode={result['mode']}, fruit_count={result['fruit_count']}, "
            f"fruits={[(f.get('id'), f.get('food'), f.get('freshness_score'), f.get('status'), f.get('bbox')) for f in result['fruits']]}"
        )
        return result
    except Exception as e:                           # never let a model problem break the app
        _detection_log(f"model analysis failed; using whole-image heuristic fallback: {e}")
        return _fallback_result(rgb_in, lighting, normalize, f"Model error: {e}")


def _primary_fruit(result):
    """The fruit to use when comparing photos: the largest one."""
    if not result.get("fruits"):
        return None
    return max(result["fruits"], key=lambda f: f["bbox"][2] * f["bbox"][3])


def compare_images(bytes_a, bytes_b, normalize=False):
    """Compare two photos of the same fruit (e.g. today vs three days later)."""
    a = analyze_image(bytes_a, normalize=normalize)
    b = analyze_image(bytes_b, normalize=normalize)
    fa, fb = _primary_fruit(a), _primary_fruit(b)

    notes = []
    for label, r in (("A", a), ("B", b)):
        if r["fruit_count"] > 1:
            notes.append(f"Photo {label} has {r['fruit_count']} fruits - the largest one was compared.")

    same_fruit, same_fruit_type, diff = False, None, None
    warning = None
    summary_a = _comparison_summary(a, fa)
    summary_b = _comparison_summary(b, fb)
    score_a, score_b = summary_a["freshness_score"], summary_b["freshness_score"]
    if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)):
        diff = round(score_b - score_a, 1)
    higher = "B" if diff is not None and diff > 0 else "A" if diff is not None and diff < 0 else "tie" if diff == 0 else None

    if a.get("analysis_blocked") or b.get("analysis_blocked"):
        dark = [l for l, r in (("A", a), ("B", b)) if r.get("analysis_blocked")]
        verdict = f"Photo {' and '.join(dark)} is too dark to analyse, so there is nothing to compare. Please retake it in better light."
    elif a["is_fruit"] is False or b["is_fruit"] is False:
        bad = [l for l, r in (("A", a), ("B", b)) if r["is_fruit"] is False]
        verdict = f"Photo {' and '.join(bad)} does not look like a fruit FreshSense knows, so there is nothing to compare."
    elif fa is None or fb is None or a["mode"] != "trained_model" or b["mode"] != "trained_model":
        verdict = ("The trained model was not available, so the fruit could not be identified. "
                   "Colour-based scores are only a rough guide.")
    else:
        same_fruit = fa["food"] == fb["food"]
        same_fruit_type = same_fruit
        if not same_fruit:
            warning = (f"The photos were predicted as different fruit types ({fa['food']} and {fb['food']}). "
                       "Their score difference may not be meaningful; choose photos of the same fruit type.")
            verdict = "FreshSense predicted different fruit types, so treat their score comparison with caution."
        else:
            if diff == 0:
                verdict = f"Both photos show {fa['food']} and received the same freshness score."
            elif higher == "A":
                verdict = f"Both photos show {fa['food']}. Photo A received the higher freshness score by {abs(diff):.1f} points."
            else:
                verdict = f"Both photos show {fa['food']}. Photo B received the higher freshness score by {abs(diff):.1f} points."
            if abs(diff) < 5 and diff != 0:
                verdict += " The scores are close."
    if notes:
        verdict += " " + " ".join(notes)

    for r in (a, b):                                  # keep the compare response small
        r.pop("annotated_image", None)
    return {"photo_a": a, "photo_b": b, "photo_a_summary": summary_a, "photo_b_summary": summary_b,
            "same_fruit": same_fruit, "same_fruit_type": same_fruit_type, "warning": warning,
            "higher_freshness": higher, "score_difference_b_minus_a": diff, "verdict": verdict,
            "status_basis": STATUS_BASIS, "status_thresholds": STATUS_THRESHOLDS,
            "disclaimer": RESULT_DISCLAIMER}


def _comparison_summary(result, fruit):
    """Return the specific fruit used for comparison, or the top-level estimate."""
    if fruit:
        return {key: fruit.get(key) for key in ("food", "freshness_score", "status", "status_icon",
                                                 "score_basis", "status_basis", "identification_warning")}
    return {key: result.get(key) for key in ("food", "freshness_score", "status", "status_icon",
                                              "score_basis", "status_basis", "identification_warning")}
