"""Tests for FreshSense score bands and honest result wording."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import freshness_model as fm


# The checks in this file are about the original, un-softened score formula, so the
# score softening (see freshness_model.soften_score) is switched off for them.
_SAVED_SOFTENING = fm.SCORE_SOFTENING


def setUpModule():
    fm.SCORE_SOFTENING = 1.0


def tearDownModule():
    fm.SCORE_SOFTENING = _SAVED_SOFTENING


class StatusBandTests(unittest.TestCase):
    def test_status_threshold_boundaries(self):
        self.assertEqual(fm._status_from_score(39.9)[0], "Spoiled / Not Fresh")
        self.assertEqual(fm._status_from_score(40)[0], "Moderate")
        self.assertEqual(fm._status_from_score(69.9)[0], "Moderate")
        self.assertEqual(fm._status_from_score(70)[0], "Fresh")

    def test_binary_model_outputs_produce_score_band_not_moderate_class(self):
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        with patch.object(fm, "_labels", labels):
            result = fm._interpret(np.array([0.42, 0.28, 0.15, 0.15]))

        self.assertEqual(result["food"], "Apple")
        self.assertEqual(result["score"], 60.0)
        self.assertEqual(result["fresh_model_score"], 0.6)
        self.assertAlmostEqual(result["rotten_model_score"], 0.4)
        self.assertEqual(result["raw_class_scores"], {"apple_fresh": 0.42, "apple_rotten": 0.28})
        self.assertNotIn("fresh_probability", result)
        self.assertNotIn("rotten_probability", result)

        item = fm._fruit_item(1, (0, 0, 8, 8), result, 0.9)
        self.assertEqual(item["status"], "Moderate")
        self.assertEqual(item["status_basis"], "experimental_score_band")
        self.assertIn("not calibrated real-world probabilities", item["details"]["model_score_note"])
        self.assertNotIn("fresh_probability", item["details"])

    def test_selected_fruit_uses_metadata_index_pair_for_raw_outputs_and_score(self):
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        outputs = np.array([0.05, 0.15, 0.6064, 0.1936])
        with patch.object(fm, "_labels", labels):
            result = fm._interpret(outputs)
            item = fm._fruit_item(1, (0, 0, 8, 8), result, 0.95)

        self.assertEqual(result["food"], "Banana")
        self.assertEqual(result["score"], 75.8)
        self.assertEqual(result["raw_class_scores"], {
            "banana_fresh": 0.6064, "banana_rotten": 0.1936,
        })
        self.assertEqual(item["status"], "Fresh")
        self.assertEqual(item["details"]["raw_class_scores"], {
            "banana_fresh": 0.6064, "banana_rotten": 0.1936,
        })
        self.assertIn("not calibrated real-world probabilities", item["details"]["model_score_note"])

    def test_real_metadata_preserves_output_class_index_mapping(self):
        metadata_path = Path(fm.__file__).resolve().parent / "model" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = fm._labels_from_metadata(metadata)

        self.assertEqual(len(labels), 16)
        self.assertEqual(labels, [f"{row['fruit']}_{row['condition']}" for row in metadata["classes"]])
        self.assertEqual(labels[0:4], ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"])
        self.assertEqual(labels[10:12], ["orange_fresh", "orange_rotten"])

    def test_expected_detector_fruit_pair_sets_score_even_when_guava_is_top(self):
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten",
                  "grape_fresh", "grape_rotten", "guava_fresh", "guava_rotten",
                  "jujube_fresh", "jujube_rotten", "orange_fresh", "orange_rotten",
                  "pomegranate_fresh", "pomegranate_rotten", "strawberry_fresh", "strawberry_rotten"]
        outputs = np.zeros(16, dtype=np.float32)
        outputs[labels.index("apple_fresh")] = 0.073883
        outputs[labels.index("apple_rotten")] = 0.197597
        outputs[labels.index("guava_fresh")] = 0.60
        outputs[labels.index("guava_rotten")] = 0.12852

        with patch.object(fm, "_labels", labels):
            result = fm._interpret(outputs, expected_food="Apple")

        self.assertEqual(result["food"], "Apple")
        self.assertEqual(result["classifier_top_food"], "Guava")
        self.assertEqual(result["score"], 27.2)
        self.assertAlmostEqual(result["score_numerator"], 0.073883, places=6)
        self.assertAlmostEqual(result["score_denominator"], 0.27148, places=6)
        self.assertEqual(set(result["all_class_scores"]), set(labels))

    def test_inference_uses_rgb_raw_pixels_and_training_bilinear_resize(self):
        class CapturingModel:
            def predict(self, batch, verbose=0):
                self.batch = batch.copy()
                return np.asarray([[0.7, 0.3]], dtype=np.float32)

        model = CapturingModel()
        rgb = np.asarray([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [200, 100, 50]],
        ], dtype=np.uint8)
        expected = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        with patch.object(fm, "_try_load_model", return_value=model), \
             patch.object(fm, "_image_size", (4, 4)), \
             patch.object(fm, "_rescale", False), \
             patch.object(fm, "_feat_model", None):
            fm._run_model([rgb])

        np.testing.assert_array_equal(model.batch[0], expected)
        self.assertGreater(float(model.batch.max()), 1.0)  # model's own rescaling layer gets 0..255

    def test_model_result_identifies_experimental_band_and_limits(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        fruit = {
            "id": 1, "bbox": [0, 0, 8, 8], "food": "Apple", "freshness_score": 60.0,
            "status": "Moderate", "status_icon": "yellow", "fruit_model_score": 0.7,
            "score_basis": "model_output_score", "status_basis": fm.STATUS_BASIS, "details": {},
        }
        with patch.object(fm, "_annotate", return_value="image"):
            result = fm._build_result(rgb, [fruit], [], "whole-image", {}, False)

        self.assertEqual(result["status_basis"], "experimental_score_band")
        self.assertEqual(result["score_basis"], "model_output_score")
        self.assertIn("have not been scientifically validated", result["disclaimer"].lower())
        self.assertIn("cannot establish food safety", result["disclaimer"])
        self.assertIn("Moderate score band", result["message"])
        self.assertNotIn("is moderate", result["message"].lower())

    def test_heuristic_fallback_is_identified_as_visual_estimate(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch.object(fm, "_heuristic_freshness", return_value=(60.0, "Moderate", "yellow", {})), \
             patch.object(fm, "_annotate", return_value="image"):
            result = fm._fallback_result(rgb, {}, False, "No trained model found")

        self.assertEqual(result["mode"], "fallback_heuristic")
        self.assertEqual(result["score_basis"], "heuristic_visual_features")
        self.assertIsNone(result["fruit_model_score"])
        self.assertIn("Heuristic visual estimate", result["message"])
        self.assertIn("not establish food safety", result["message"])
        self.assertIn("Moderate is not a class", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
