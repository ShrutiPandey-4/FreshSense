"""Tests for the three-classifier comparison pipeline and endpoint."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as api
import model_comparison as comparison


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(output, format="PNG")
    return output.getvalue()


class ModelMetadataTests(unittest.TestCase):
    def test_metadata_class_order_is_preserved(self):
        labels = comparison._metadata_classes({
            "classes": [
                {"fruit": "banana", "condition": "rotten"},
                {"fruit": "apple", "condition": "fresh"},
            ],
        })

        self.assertEqual(labels, ["banana_rotten", "apple_fresh"])

    def test_research_metadata_class_names_and_preprocessing_are_respected(self):
        efficientnet = {
            "architecture": "EfficientNetB0",
            "class_names": ["FreshApple", "FreshBanana", "RottenApple", "RottenBanana"],
        }
        resnet = {
            "class_names": ["FreshApple", "RottenApple"],
            "preprocessing": "ResNet50 preprocess_input embedded in model",
        }

        self.assertEqual(
            comparison._metadata_classes(efficientnet),
            ["apple_fresh", "banana_fresh", "apple_rotten", "banana_rotten"],
        )
        self.assertEqual(comparison._preprocessing(efficientnet), "raw_0_255")
        self.assertEqual(comparison._preprocessing(resnet), "raw_0_255")

    def test_freshness_status_is_derived_from_binary_outputs(self):
        labels = ["banana_rotten", "apple_fresh", "banana_fresh", "apple_rotten"]
        result = comparison._interpret(np.asarray([0.1, 0.6, 0.1, 0.2]), labels)

        self.assertEqual(result["predicted_fruit"], "Apple")
        self.assertEqual(result["prediction"], "Fresh")
        self.assertEqual(result["freshness_score"], 63.4)
        self.assertEqual(result["freshness_status"], "Moderate")

    def test_resnet50_preprocessing_uses_keras_rgb_to_bgr_transform(self):
        rgb = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
        batch = comparison._preprocess(rgb, (1, 1), "resnet50")

        np.testing.assert_allclose(
            batch[0, 0, 0],
            [3 - 103.939, 2 - 116.779, 1 - 123.68],
            atol=1e-3,
        )

    def test_missing_research_model_is_reported_as_unavailable(self):
        key = "efficientnetb0"
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "name": "EfficientNetB0",
                "path": os.path.join(temp_dir, "fruit_condition.keras"),
                "metadata_path": os.path.join(temp_dir, "metadata.json"),
            }
            comparison._research_loads.pop(key, None)
            with patch.dict(comparison._MODEL_CONFIG, {key: config}):
                result = comparison._load_research_model(key)
            comparison._research_loads.pop(key, None)

        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["message"], "Model not available")

    def test_same_decoded_image_is_supplied_to_all_model_runners(self):
        rgb = np.full((8, 10, 3), 127, dtype=np.uint8)
        received = []

        def capture(_key, image):
            received.append(image)
            return {"state": "success"}

        with patch.object(comparison.fm, "_read_image", return_value=rgb), \
             patch.object(comparison, "_mobilenet_result", side_effect=lambda image: capture("mobilenetv2", image)), \
             patch.object(comparison, "_research_result", side_effect=lambda key, image: capture(key, image)):
            result = comparison.compare_classification_models(b"one image")

        self.assertEqual(len(received), 3)
        self.assertTrue(all(image is rgb for image in received))
        self.assertEqual(len(result["models"]), 3)

    def test_research_prediction_uses_metadata_index_mapping(self):
        labels = ["banana_rotten", "apple_fresh", "banana_fresh", "apple_rotten"]

        class FakeModel:
            def predict(self, batch, verbose=0):
                self.batch = batch
                return np.asarray([[0.1, 0.6, 0.1, 0.2]])

        model = FakeModel()
        loaded = {
            "state": "success",
            "model": model,
            "labels": labels,
            "preprocessing": "raw_0_255",
            "image_size": (4, 4),
        }
        with patch.object(comparison, "_load_research_model", return_value=loaded):
            result = comparison._research_result("resnet50", np.full((4, 4, 3), 50, dtype=np.uint8))

        self.assertEqual(result["state"], "success")
        self.assertEqual(result["predicted_fruit"], "Apple")
        self.assertEqual(result["prediction"], "Fresh")
        self.assertEqual(model.batch.shape, (1, 4, 4, 3))

    def test_missing_research_model_is_reported_without_simulated_predictions(self):
        with patch.object(comparison, "_load_research_model", return_value={
            "state": "unavailable", "message": "Model not available",
        }):
            result = comparison._research_result("resnet50", np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["message"], "Model not available")
        self.assertEqual(result["availability"], "Unavailable")
        self.assertIsNone(result["freshness_score"])
        self.assertIsNone(result["inference_time_ms"])

    def test_supplied_research_models_load_and_predict(self):
        image_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "banana_plate_failure.png")
        with Image.open(image_path) as image_file:
            rgb = np.asarray(image_file.convert("RGB"))
        expected_labels = [
            "apple_fresh", "banana_fresh", "grape_fresh", "guava_fresh",
            "jujube_fresh", "orange_fresh", "pomegranate_fresh", "strawberry_fresh",
            "apple_rotten", "banana_rotten", "grape_rotten", "guava_rotten",
            "jujube_rotten", "orange_rotten", "pomegranate_rotten", "strawberry_rotten",
        ]

        for key in ("efficientnetb0", "resnet50"):
            with self.subTest(model=key):
                comparison._research_loads.pop(key, None)
                loaded = comparison._load_research_model(key)
                self.assertEqual(loaded["state"], "success", loaded.get("message"))
                self.assertEqual(loaded["labels"], expected_labels)
                self.assertEqual(loaded["image_size"], (224, 224))
                self.assertEqual(loaded["preprocessing"], "raw_0_255")
                result = comparison._research_result(key, rgb)
                self.assertEqual(result["state"], "success", result.get("message"))
                self.assertIsNotNone(result["predicted_fruit"])
                self.assertIn(result["prediction"], ("Fresh", "Rotten"))
                self.assertGreaterEqual(result["freshness_score"], 2.0)
                self.assertLessEqual(result["freshness_score"], 98.0)
                self.assertGreaterEqual(result["inference_time_ms"], 0)


class CompareModelsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def test_valid_upload_returns_json_and_passes_same_image_once(self):
        expected = {"models": [{"model_name": "MobileNetV2", "state": "unavailable"}]}
        with patch.object(api, "compare_classification_models", return_value=expected) as compare:
            response = self.client.post(
                "/api/compare-models",
                data={"image": (io.BytesIO(png_bytes()), "fruit.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(compare.call_count, 1)
        self.assertEqual(compare.call_args.args[0], png_bytes())

    def test_invalid_image_upload_is_rejected(self):
        response = self.client.post(
            "/api/compare-models",
            data={"image": (io.BytesIO(b"not an image"), "fruit.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid or supported image", response.get_json()["error"])

    def test_existing_analyze_and_compare_endpoints_still_work(self):
        with patch.object(api, "analyze_image", return_value={"fruit_count": 1}) as analyze:
            analyze_response = self.client.post(
                "/api/analyze",
                data={"image": (io.BytesIO(png_bytes()), "fruit.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(analyze_response.status_code, 200)
        self.assertEqual(analyze_response.get_json(), {"fruit_count": 1})
        analyze.assert_called_once()

        with patch.object(api, "compare_images", return_value={"same_fruit": True}) as compare:
            compare_response = self.client.post(
                "/api/compare",
                data={
                    "image_a": (io.BytesIO(png_bytes()), "a.png"),
                    "image_b": (io.BytesIO(png_bytes()), "b.png"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(compare_response.status_code, 200)
        self.assertEqual(compare_response.get_json(), {"same_fruit": True})
        compare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
