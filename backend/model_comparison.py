"""Inference helpers for comparing the three fruit-condition classifiers."""

import json
import logging
import os
import threading
import time

import cv2
import numpy as np

import freshness_model as fm

_HERE = os.path.dirname(os.path.abspath(__file__))


def _research_model_paths(key):
    project_root = os.path.dirname(_HERE)
    candidates = (
        os.path.join(project_root, "research_models", key),
        os.path.join(project_root, "model", "researchmodels", key),
    )
    model_dir = next((path for path in candidates if os.path.isdir(path)), candidates[0])
    return (
        os.path.join(model_dir, "fruit_condition.keras"),
        os.path.join(model_dir, "metadata.json"),
    )


_EFFICIENTNET_PATH, _EFFICIENTNET_METADATA = _research_model_paths("efficientnetb0")
_RESNET_PATH, _RESNET_METADATA = _research_model_paths("resnet50")
_MODEL_CONFIG = {
    "mobilenetv2": {
        "name": "MobileNetV2",
        "path": fm._KERAS_MODEL_PATH,
        "metadata_path": fm._METADATA_PATH,
    },
    "efficientnetb0": {
        "name": "EfficientNetB0",
        "path": _EFFICIENTNET_PATH,
        "metadata_path": _EFFICIENTNET_METADATA,
    },
    "resnet50": {
        "name": "ResNet50",
        "path": _RESNET_PATH,
        "metadata_path": _RESNET_METADATA,
    },
}
_research_loads = {}
_research_load_lock = threading.Lock()


def _metadata_classes(metadata):
    """Return labels in saved output-index order."""
    classes = metadata.get("classes", metadata.get("class_names"))
    if not isinstance(classes, list) or not classes:
        raise ValueError("metadata must include an ordered 'classes' list")

    labels = []
    for item in classes:
        if isinstance(item, dict):
            fruit = item.get("fruit")
            condition = item.get("condition")
            if not isinstance(fruit, str) or not isinstance(condition, str):
                raise ValueError("each metadata class needs fruit and condition values")
            labels.append(f"{fruit}_{condition}".strip().lower())
        elif isinstance(item, str) and "_" in item:
            labels.append(item.strip().lower())
        elif isinstance(item, str):
            normalized = item.strip().lower()
            condition = next(
                (value for value in ("fresh", "rotten", "spoiled") if normalized.startswith(value)),
                None,
            )
            fruit = normalized[len(condition):] if condition else ""
            if not fruit:
                raise ValueError("metadata classes must use '<fruit>_<condition>' labels")
            labels.append(f"{fruit}_{condition}")
        else:
            raise ValueError("metadata classes must use '<fruit>_<condition>' labels")
    return labels


def _preprocessing(metadata):
    value = metadata.get("preprocessing", metadata.get("input"))
    if not isinstance(value, str) or not value.strip():
        if str(metadata.get("architecture", "")).lower() == "efficientnetb0":
            return "raw_0_255"
        raise ValueError("metadata must declare the model's input preprocessing")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")

    if "resnet50" in normalized:
        return "raw_0_255" if "embedded" in normalized else "resnet50"
    if "efficientnetb0" in normalized:
        return "raw_0_255"
    if "mobilenetv2" in normalized:
        return "mobilenetv2"
    if normalized in {"raw", "raw_0_255", "rgb_0_255", "rgb_pixels_0..255", "rgb_pixels_0_255"}:
        return "raw_0_255"
    if normalized in {"rescale_0_1", "0..1", "0_1", "divide_by_255"}:
        return "rescale_0_1"
    if "0..255" in normalized or "0_255" in normalized:
        return "raw_0_255"
    raise ValueError(f"unsupported preprocessing declaration: {value}")


def _load_research_model(key):
    """Load a research model and metadata once; retain failures until restart."""
    with _research_load_lock:
        return _load_research_model_once(key)


def _load_research_model_once(key):
    if key in _research_loads:
        return _research_loads[key]

    config = _MODEL_CONFIG[key]
    if not os.path.isfile(config["path"]):
        loaded = {"state": "unavailable", "message": "Model not available"}
    elif not os.path.isfile(config["metadata_path"]):
        loaded = {"state": "error", "message": "Model metadata not available"}
    else:
        try:
            with open(config["metadata_path"], encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            labels = _metadata_classes(metadata)
            preprocessing = _preprocessing(metadata)
            image_size = metadata.get("image_size", metadata.get("input_size", [224, 224]))
            if (not isinstance(image_size, list) or len(image_size) not in (2, 3)
                    or any(not isinstance(size, int) or size <= 0 for size in image_size)):
                raise ValueError("metadata image_size must declare positive height and width")
            if len(image_size) == 3 and image_size[2] != 3:
                raise ValueError("metadata input_size must declare three RGB channels")
            if metadata.get("num_classes", len(labels)) != len(labels):
                raise ValueError("metadata num_classes does not match its class names")

            import tensorflow as tf

            custom_objects = None
            if key == "resnet50":
                custom_objects = {
                    "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
                }
            model = tf.keras.models.load_model(
                config["path"],
                compile=False,
                custom_objects=custom_objects,
            )
            output_shape = model.output_shape
            if (not isinstance(output_shape, tuple) or len(output_shape) != 2
                    or output_shape[-1] != len(labels)):
                raise ValueError("model output does not match its metadata classes")
            loaded = {
                "state": "success",
                "model": model,
                "labels": labels,
                "preprocessing": preprocessing,
                "image_size": tuple(image_size[:2]),
            }
        except Exception:
            logging.getLogger(__name__).exception("Could not load %s comparison model", config["name"])
            loaded = {"state": "error", "message": "Model could not be loaded"}

    _research_loads[key] = loaded
    return loaded


def _preprocess(rgb, image_size, preprocessing):
    height, width = image_size
    batch = np.expand_dims(
        cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype("float32"),
        axis=0,
    )
    if preprocessing == "raw_0_255" or preprocessing == "efficientnetb0":
        return batch

    import tensorflow as tf

    if preprocessing == "rescale_0_1":
        return batch / 255.0
    if preprocessing == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(batch)
    if preprocessing == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    raise ValueError(f"unsupported preprocessing declaration: {preprocessing}")


def _probabilities(output, class_count):
    values = np.asarray(output)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.size != class_count:
        raise ValueError("model output does not match its metadata classes")
    values = values.astype("float64")
    if not np.all(np.isfinite(values)):
        raise ValueError("model returned non-finite class scores")
    total = float(values.sum())
    if (np.any(values < 0) or np.any(values > 1)
            or not np.isclose(total, 1.0, rtol=1e-3, atol=1e-3)):
        values = np.exp(values - np.max(values))
        total = float(values.sum())
    if total <= 0:
        raise ValueError("model returned empty class scores")
    return values / total


def _interpret(probabilities, labels):
    grouped = {}
    for label, score in zip(labels, probabilities):
        if "_" not in label:
            continue
        fruit, condition = label.rsplit("_", 1)
        if condition in {"fresh", "rotten", "spoiled"}:
            grouped.setdefault(fruit, {})[condition] = float(score)

    fruit_scores = {
        fruit: scores.get("fresh", 0.0) + scores.get("rotten", 0.0) + scores.get("spoiled", 0.0)
        for fruit, scores in grouped.items()
        if scores.get("fresh", 0.0) + scores.get("rotten", 0.0) + scores.get("spoiled", 0.0) > 0
    }
    if not fruit_scores:
        raise ValueError("model metadata has no fresh and rotten class pairs")

    fruit = max(fruit_scores, key=fruit_scores.get)
    scores = grouped[fruit]
    fresh_score = scores.get("fresh", 0.0)
    rotten_score = scores.get("rotten", 0.0) + scores.get("spoiled", 0.0)
    denominator = fresh_score + rotten_score
    if denominator <= 0:
        raise ValueError("predicted fruit has no fresh or rotten class scores")

    freshness_score = round(fm.soften_score(fresh_score / denominator), 1)
    freshness_status, status_icon = fm._status_from_score(freshness_score)
    return {
        "predicted_fruit": fruit.replace("_", " ").title(),
        "prediction": "Fresh" if fresh_score >= rotten_score else "Rotten",
        "freshness_score": freshness_score,
        "freshness_status": freshness_status,
        "status_icon": status_icon,
    }


def _unavailable_result(config, state, message):
    return {
        "model_name": config["name"],
        "state": state,
        "availability": "Unavailable" if state == "unavailable" else "Error",
        "message": message,
        "predicted_fruit": None,
        "prediction": None,
        "freshness_score": None,
        "freshness_status": None,
        "inference_time_ms": None,
    }


def _research_result(key, rgb):
    config = _MODEL_CONFIG[key]
    loaded = _load_research_model(key)
    if loaded["state"] != "success":
        return _unavailable_result(config, loaded["state"], loaded["message"])

    try:
        started = time.perf_counter()
        batch = _preprocess(rgb, loaded["image_size"], loaded["preprocessing"])
        output = loaded["model"].predict(batch, verbose=0)
        inference_time_ms = round((time.perf_counter() - started) * 1000, 2)
        probabilities = _probabilities(output, len(loaded["labels"]))
        prediction = _interpret(probabilities, loaded["labels"])
        return {
            "model_name": config["name"],
            "state": "success",
            "availability": "Available",
            "inference_time_ms": inference_time_ms,
            **prediction,
        }
    except Exception:
        logging.getLogger(__name__).exception("Prediction failed for %s comparison model", config["name"])
        return _unavailable_result(config, "error", "Prediction failed")


def _mobilenet_result(rgb):
    config = _MODEL_CONFIG["mobilenetv2"]
    try:
        if fm._try_load_model() is None or not fm._labels:
            model_files_exist = os.path.isfile(config["path"]) and os.path.isfile(config["metadata_path"])
            state = "error" if model_files_exist else "unavailable"
            message = "Model could not be loaded" if model_files_exist else "Model not available"
            return _unavailable_result(config, state, message)
        started = time.perf_counter()
        probabilities, _, _ = fm._run_model([rgb])
        inference_time_ms = round((time.perf_counter() - started) * 1000, 2)
        prediction = _interpret(
            _probabilities(probabilities[0], len(fm._labels)),
            fm._labels,
        )
        return {
            "model_name": config["name"],
            "state": "success",
            "availability": "Available",
            "inference_time_ms": inference_time_ms,
            **prediction,
        }
    except Exception:
        logging.getLogger(__name__).exception("Prediction failed for MobileNetV2 comparison model")
        return _unavailable_result(config, "error", "Prediction failed")


def compare_classification_models(image_bytes, normalize=False):
    """Run all available classifiers against the same uploaded RGB image."""
    rgb = fm._read_image(image_bytes)
    if normalize:
        rgb = fm.normalize_lighting(rgb)
    return {
        "models": [
            _mobilenet_result(rgb),
            _research_result("efficientnetb0", rgb),
            _research_result("resnet50", rgb),
        ],
        "note": "Only trained-model predictions are shown. Fresh, Moderate, and Spoiled are score bands; Moderate is not a trained class.",
    }
