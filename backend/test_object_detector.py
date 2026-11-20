"""Focused tests for the pretrained detector to existing classifier handoff."""

import unittest
from unittest.mock import patch

import numpy as np

import freshness_model as fm


# The checks in this file are about the original, un-softened score formula, so the
# score softening (see freshness_model.soften_score) is switched off for them.
_SAVED_SOFTENING = fm.SCORE_SOFTENING


def setUpModule():
    fm.SCORE_SOFTENING = 1.0


def tearDownModule():
    fm.SCORE_SOFTENING = _SAVED_SOFTENING


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeDetection:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [self]
        self._xyxy = xyxy

    def tolist(self):
        return self._xyxy


class FakeOutput:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    names = {46: "banana", 47: "apple", 49: "orange", 53: "pizza"}

    def __init__(self, rows):
        self.rows = rows
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [FakeOutput(self.rows)]


class ObjectDetectorPipelineTests(unittest.TestCase):
    def test_crop_uses_small_padding_and_preserves_rectangular_dimensions(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        box = (10, 20, 20, 40)

        crop = fm._crop_region(image, box, pad_frac=0.10)

        self.assertEqual(crop.shape, (48, 24, 3))
        self.assertEqual(box, (10, 20, 20, 40))  # reporting box is never modified

    def test_crop_padding_is_clamped_at_image_boundaries(self):
        image = np.zeros((30, 40, 3), dtype=np.uint8)

        crop = fm._crop_region(image, (-4, -3, 12, 10), pad_frac=0.10)

        self.assertEqual(crop.shape, (8, 9, 3))
        self.assertLessEqual(crop.shape[0], image.shape[0])
        self.assertLessEqual(crop.shape[1], image.shape[1])

    def test_crop_padding_is_configurable_but_capped(self):
        with patch.dict("os.environ", {"FRESHSENSE_CROP_PADDING": "0.08"}):
            self.assertEqual(fm._crop_padding_fraction(), 0.08)
        with patch.dict("os.environ", {"FRESHSENSE_CROP_PADDING": "0.50"}):
            self.assertEqual(fm._crop_padding_fraction(), fm.MAX_CROP_PADDING)

    def test_crop_does_not_extend_into_a_nearby_detected_fruit(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[30:70, 20:60] = (255, 0, 0)
        image[30:70, 62:102] = (0, 255, 0)
        target = (20, 30, 40, 40)
        nearby = (62, 30, 40, 40)

        crop = fm._crop_region(image, target, pad_frac=0.10, nearby_boxes=[target, nearby])

        self.assertEqual(crop.shape, (48, 46, 3))  # x=16..62; green fruit starts at x=62
        self.assertFalse(np.any(np.all(crop == (0, 255, 0), axis=2)))

    def test_detector_keeps_only_apple_banana_and_orange_and_bounds_boxes(self):
        detector = FakeDetector([
            FakeDetection(47, 0.91, [10, 20, 100, 120]),
            FakeDetection(46, 0.87, [-5, 30, 60, 140]),
            FakeDetection(53, 0.99, [120, 10, 200, 90]),
            FakeDetection(49, 0.10, [20, 20, 80, 80]),
        ])
        rgb = np.zeros((160, 180, 3), dtype=np.uint8)
        with patch.object(fm, "_load_detector", return_value=detector):
            detections = fm._detect_supported_fruits(rgb)

        self.assertEqual([item["food"] for item in detections], ["Apple", "Banana"])
        self.assertEqual(detections[0]["bbox"], (10, 20, 90, 100))
        self.assertEqual(detections[1]["bbox"], (0, 30, 60, 110))
        self.assertEqual(detections[0]["confidence"], 0.91)
        self.assertEqual(detector.kwargs["device"], "cpu")
        self.assertEqual(detector.kwargs["conf"], 0.001)
        self.assertNotIn("classes", detector.kwargs)

    def test_detector_crop_results_keep_confidence_separate_from_freshness_score(self):
        image = np.full((200, 300, 3), 230, dtype=np.uint8)
        detections = [
            {"bbox": (10, 20, 90, 100), "food": "Apple", "confidence": 0.93},
            {"bbox": (150, 30, 100, 130), "food": "Banana", "confidence": 0.81},
        ]
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten",
                  "orange_fresh", "orange_rotten"]
        model_scores = np.asarray([
            [0.75, 0.05, 0.08, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.70, 0.10, 0.08, 0.04],
        ], dtype=np.float32)

        with patch.object(fm, "_labels", labels), \
             patch.object(fm, "_run_model", return_value=(model_scores, np.array([1.0, 1.0]), "max_softmax")), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_detected_regions(image, detections, {}, False)

        self.assertEqual(result["detection_source"], "yolo11n_coco_detector")
        self.assertEqual(result["fruit_count"], 2)
        self.assertEqual([f["food"] for f in result["fruits"]], ["Apple", "Banana"])
        self.assertEqual([f["bbox"] for f in result["fruits"]], [
            [10, 20, 90, 100], [150, 30, 100, 130],
        ])
        self.assertEqual([f["detector_confidence"] for f in result["fruits"]], [0.93, 0.81])
        self.assertEqual([f["freshness_score"] for f in result["fruits"]], [93.7, 87.5])
        self.assertNotEqual(result["fruits"][0]["detector_confidence"],
                            result["fruits"][0]["freshness_score"])

    def test_detector_label_is_used_for_condition_pair_and_disagreement_is_reported(self):
        image = np.full((180, 180, 3), 200, dtype=np.uint8)
        detection = [{"bbox": (20, 20, 120, 120), "food": "Apple", "confidence": 0.88}]
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        # Banana is the classifier's top fruit, but the detector says this crop is an apple.
        model_scores = np.asarray([[0.10, 0.10, 0.65, 0.15]], dtype=np.float32)

        with patch.object(fm, "_labels", labels), \
             patch.object(fm, "_run_model", return_value=(model_scores, np.array([1.0]), "max_softmax")), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_detected_regions(image, detection, {}, False)

        fruit = result["fruits"][0]
        self.assertEqual(fruit["food"], "Apple")
        self.assertEqual(fruit["freshness_score"], 50.0)
        self.assertEqual(fruit["detector_confidence"], 0.88)
        self.assertEqual(fruit["classifier_food_prediction"], "Banana")
        self.assertIn("Fruit identification uncertain", fruit["identification_warning"])
        self.assertIn("classifier's top fruit label was Banana", fruit["note"])

    def test_frontend_renders_identification_warning_and_experimental_score_label(self):
        frontend_path = fm.os.path.join(fm.os.path.dirname(fm.__file__), "..", "frontend", "index.html")
        with open(frontend_path, encoding="utf-8") as frontend_file:
            frontend = frontend_file.read()

        self.assertIn("f.identification_warning", frontend)
        self.assertIn("Experimental model score", frontend)

    def test_unavailable_detector_is_marked_when_contour_fallback_runs(self):
        image = np.full((180, 180, 3), 230, dtype=np.uint8)
        labels = ["apple_fresh", "apple_rotten"]
        predictions = np.asarray([
            [0.8, 0.1],  # whole image
            [0.7, 0.2],  # contour crop
        ], dtype=np.float32)

        with patch.object(fm, "looks_like_graphic", return_value=False), \
             patch.object(fm, "_detect_supported_fruits", return_value=None), \
             patch.object(fm, "find_fruit_regions", return_value=(
                 [(30, 20, 100, 110)], "plain-background"
             )), \
             patch.object(fm, "_labels", labels), \
             patch.object(fm, "_run_model", return_value=(
                 predictions, np.array([1.0, 1.0]), "max_softmax"
             )), \
             patch.object(fm, "gate_threshold", return_value=0.0), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_with_model(image, {}, False)

        self.assertEqual(result["fruit_count"], 1)
        self.assertEqual(result["detection_source"], "opencv_contour_fallback")
        self.assertIn("YOLO detector unavailable", result["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
