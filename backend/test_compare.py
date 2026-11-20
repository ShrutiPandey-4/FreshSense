"""Focused tests for the existing two-photo comparison flow."""

import io
import unittest
from unittest.mock import patch

from PIL import Image

import app as api
from freshness_model import compare_images


def result_for(food, score):
    status = "Fresh" if score >= 70 else "Moderate" if score >= 40 else "Spoiled / Not Fresh"
    fruit = {
        "food": food,
        "freshness_score": score,
        "status": status,
        "status_icon": "*",
        "bbox": [0, 0, 20, 20],
    }
    return {
        "mode": "trained_model",
        "is_fruit": True,
        "fruit_count": 1,
        "fruits": [fruit],
        "food": food,
        "freshness_score": score,
        "status": status,
        "status_icon": "*",
        "annotated_image": "unused in comparison response",
    }


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(output, format="PNG")
    return output.getvalue()


class CompareImagesTests(unittest.TestCase):
    @patch("freshness_model.analyze_image")
    def test_successful_comparison_reports_each_prediction_and_higher_score(self, analyze):
        analyze.side_effect = [result_for("Apple", 82), result_for("Apple", 61)]

        data = compare_images(b"photo a", b"photo b")

        self.assertEqual(analyze.call_count, 2)
        self.assertTrue(data["same_fruit_type"])
        self.assertEqual(data["photo_a_summary"]["food"], "Apple")
        self.assertEqual(data["photo_b_summary"]["freshness_score"], 61)
        self.assertEqual(data["score_difference_b_minus_a"], -21)
        self.assertEqual(data["higher_freshness"], "A")
        self.assertIn("Photo A received the higher", data["verdict"])

    @patch("freshness_model.analyze_image")
    def test_mismatched_fruit_types_include_warning(self, analyze):
        analyze.side_effect = [result_for("Apple", 82), result_for("Banana", 61)]

        data = compare_images(b"photo a", b"photo b")

        self.assertFalse(data["same_fruit_type"])
        self.assertIn("different fruit types", data["warning"])
        self.assertEqual(data["score_difference_b_minus_a"], -21)
        self.assertEqual(data["higher_freshness"], "A")

    @patch("freshness_model.analyze_image")
    def test_identification_uncertainty_is_included_in_comparison_summaries(self, analyze):
        photo_a = result_for("Apple", 82)
        photo_a["fruits"][0]["identification_warning"] = (
            "Fruit identification uncertain: detector said Apple, classifier said Guava."
        )
        analyze.side_effect = [photo_a, result_for("Apple", 61)]

        data = compare_images(b"photo a", b"photo b")

        self.assertIn("Fruit identification uncertain", data["photo_a_summary"]["identification_warning"])


class CompareEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def test_successful_uploads_return_comparison_response(self):
        expected = {
            "photo_a": {}, "photo_b": {}, "photo_a_summary": {}, "photo_b_summary": {},
            "same_fruit": True, "same_fruit_type": True, "warning": None,
            "higher_freshness": "B", "score_difference_b_minus_a": 8.5, "verdict": "Photo B scored higher.",
        }
        with patch.object(api, "compare_images", return_value=expected):
            response = self.client.post("/api/compare", data={
                "image_a": (io.BytesIO(png_bytes()), "a.png"),
                "image_b": (io.BytesIO(png_bytes()), "b.png"),
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["score_difference_b_minus_a"], 8.5)

    def test_invalid_image_bytes_are_rejected_as_bad_request(self):
        response = self.client.post("/api/compare", data={
            "image_a": (io.BytesIO(b"not an image"), "a.png"),
            "image_b": (io.BytesIO(png_bytes()), "b.png"),
        }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid or supported image", response.get_json()["error"])

    def test_prediction_failure_returns_helpful_error(self):
        with patch.object(api, "compare_images", side_effect=RuntimeError("model details")), \
             patch.object(api.app.logger, "exception"):
            response = self.client.post("/api/compare", data={
                "image_a": (io.BytesIO(png_bytes()), "a.png"),
                "image_b": (io.BytesIO(png_bytes()), "b.png"),
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Could not compare these photos. Please try different images.")
        self.assertNotIn("model details", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
