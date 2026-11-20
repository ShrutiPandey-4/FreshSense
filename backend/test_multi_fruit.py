"""Regression tests for FreshSense's existing contour and per-crop flow."""

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import freshness_model as fm


def fruit_scene(circles):
    """Build a white-background scene; circles are (center_x, center_y, radius)."""
    rgb = np.full((480, 480, 3), 255, dtype=np.uint8)
    for x, y, radius in circles:
        cv2.circle(rgb, (x, y), radius, (210, 65, 35), thickness=-1)
    return rgb


def bananas_on_plate_scene():
    """Build a plate-containing scene that previously became one outer contour."""
    rgb = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.ellipse(rgb, (320, 250), (300, 190), 0, 0, 360, (220, 220, 220), -1)
    for x, y, angle, color in (
        (145, 230, -15, (30, 210, 235)),
        (320, 235, 0, (35, 190, 230)),
        (495, 230, 15, (25, 170, 225)),
    ):
        cv2.ellipse(rgb, (x, y), (85, 30), angle, 0, 360, color, -1)
    return rgb


def class_outputs(rows):
    return np.asarray(rows, dtype=np.float32)


# The checks in this file are about the original, un-softened score formula, so the
# score softening (see freshness_model.soften_score) is switched off for them.
_SAVED_SOFTENING = fm.SCORE_SOFTENING


def setUpModule():
    fm.SCORE_SOFTENING = 1.0


def tearDownModule():
    fm.SCORE_SOFTENING = _SAVED_SOFTENING


class MultiFruitRegionTests(unittest.TestCase):
    def test_real_banana_plate_label_rectangles_are_in_bounds_and_separate(self):
        image_path = Path(__file__).resolve().parents[1] / "testdata" / "banana_plate_failure.png"
        rgb = fm._read_image(image_path.read_bytes())
        boxes, _ = fm.find_fruit_regions(rgb)
        self.assertEqual(len(boxes), 3, "regression image should produce three separate fruit boxes")

        image_h, image_w = rgb.shape[:2]
        scale = 720.0 / max(image_h, image_w) if max(image_h, image_w) > 720 else 1.0
        canvas_w, canvas_h = int(image_w * scale), int(image_h * scale)
        tag_size = min(max(30, int(round(32 * scale))), canvas_w, canvas_h)
        entries = [
            (tuple(int(value * scale) for value in box),
             (tag_size, tag_size), (tag_size, tag_size))
            for box in boxes
        ]

        rects = fm._annotation_label_rectangles(canvas_w, canvas_h, entries)
        self.assertEqual(len(rects), 3)
        labels = [{"id": index, "bbox": box, "status": "Fresh"}
                  for index, box in enumerate(boxes, 1)]
        annotated = fm._annotate(rgb, labels, [])
        self.assertTrue(annotated.startswith("data:image/jpeg;base64,"))
        for x, y, width, height, _compact in rects:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, canvas_w)
            self.assertLessEqual(y + height, canvas_h)
        for index, (x, y, width, height, _compact) in enumerate(rects):
            for ox, oy, ow, oh, _other_compact in rects[index + 1:]:
                self.assertTrue(x + width <= ox or ox + ow <= x or
                                y + height <= oy or oy + oh <= y)

    def test_mixed_fruit_failure_photo_is_a_contour_regression_fixture(self):
        image_path = Path(__file__).resolve().parents[1] / "testdata" / "mixed_fruit_failure.png"
        rgb = fm._read_image(image_path.read_bytes())
        regions, mode = fm.find_fruit_regions(rgb)

        self.assertEqual(mode, "plain-background")
        # The real photo currently produces three crops, including a merged
        # multi-fruit crop. This guards the observed refinement path; it does
        # not claim that every fruit has been separated successfully.
        self.assertGreaterEqual(len(regions), 3)

    def test_three_bananas_on_plate_get_separate_boxes_and_predictions(self):
        rgb = bananas_on_plate_scene()
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        predictions = class_outputs([
            [0.20, 0.10, 0.60, 0.10],  # whole plate candidate, ignored if inner regions are found
            [0.02, 0.02, 0.91, 0.05],  # Banana: Fresh
            [0.02, 0.02, 0.51, 0.45],  # Banana: Moderate score band
            [0.02, 0.02, 0.05, 0.91],  # Banana: Spoiled / Not Fresh
        ])

        with patch.object(fm, "looks_like_graphic", return_value=False), \
             patch.object(fm, "_detect_supported_fruits", return_value=[]), \
             patch.object(fm, "_labels", labels), \
             patch.object(fm, "_run_model", return_value=(predictions, np.ones(4), "max_logit")), \
             patch.object(fm, "gate_threshold", return_value=0.0), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_with_model(rgb, {}, False)

        self.assertEqual(result["fruit_count"], 3)
        self.assertEqual(
            [(f["id"], f["food"], f["status"]) for f in result["fruits"]],
            [(1, "Banana", "Fresh"), (2, "Banana", "Moderate"),
             (3, "Banana", "Spoiled / Not Fresh")],
        )
        self.assertEqual(len({tuple(f["bbox"]) for f in result["fruits"]}), 3)
        self.assertTrue(all(f["freshness_score"] is not None for f in result["fruits"]))
        self.assertEqual(result["annotated_image"], "annotated")

    def test_separated_fruit_regions_are_predicted_independently(self):
        rgb = fruit_scene([(115, 150, 52), (335, 310, 52)])
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        predictions = class_outputs([
            [0.05, 0.05, 0.85, 0.05],  # whole-image candidate (not used when regions pass)
            [0.02, 0.02, 0.91, 0.05],  # first banana: Fresh
            [0.02, 0.02, 0.05, 0.91],  # second banana: Spoiled / Not Fresh
        ])

        with patch.object(fm, "looks_like_graphic", return_value=False), \
             patch.object(fm, "_detect_supported_fruits", return_value=[]), \
             patch.object(fm, "_labels", labels), \
             patch.object(fm, "find_fruit_regions", return_value=(
                 [(63, 98, 105, 105), (283, 258, 105, 105)], "plain-background"
             )), \
             patch.object(fm, "_run_model", return_value=(predictions, np.ones(3), "max_logit")), \
             patch.object(fm, "gate_threshold", return_value=0.0), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_with_model(rgb, {}, False)

        self.assertEqual(result["fruit_count"], 2)
        self.assertEqual(len(result["fruits"]), 2)
        self.assertEqual(
            [(f["id"], f["food"], f["status"]) for f in result["fruits"]],
            [(1, "Banana", "Fresh"), (2, "Banana", "Spoiled / Not Fresh")],
        )
        self.assertEqual([f["bbox"] for f in result["fruits"]], [
            [63, 98, 105, 105], [283, 258, 105, 105],
        ])
        self.assertEqual([f["freshness_score"] for f in result["fruits"]], [94.8, 5.2])

    def test_separated_mixed_fruits_get_distinct_types_and_conditions(self):
        rgb = fruit_scene([(115, 150, 52), (335, 310, 52)])
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        predictions = class_outputs([
            [0.05, 0.05, 0.85, 0.05],
            [0.02, 0.02, 0.91, 0.05],  # Banana
            [0.05, 0.91, 0.02, 0.02],  # Apple
        ])

        with patch.object(fm, "looks_like_graphic", return_value=False), \
             patch.object(fm, "_detect_supported_fruits", return_value=[]), \
             patch.object(fm, "_labels", labels), \
             patch.object(fm, "find_fruit_regions", return_value=(
                 [(63, 98, 105, 105), (283, 258, 105, 105)], "plain-background"
             )), \
             patch.object(fm, "_run_model", return_value=(predictions, np.ones(3), "max_logit")), \
             patch.object(fm, "gate_threshold", return_value=0.0), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm._analyze_with_model(rgb, {}, False)

        self.assertEqual(result["fruit_count"], 2)
        self.assertEqual(
            [(f["food"], f["status"]) for f in result["fruits"]],
            [("Banana", "Fresh"), ("Apple", "Spoiled / Not Fresh")],
        )
        for fruit in result["fruits"]:
            self.assertIn("freshness_score", fruit)
            self.assertEqual(len(fruit["bbox"]), 4)

    def test_small_region_below_current_area_cutoff_is_ignored(self):
        # A 40px-radius circle is about 2.2% of this frame: below the existing 2.5% filter.
        rgb = fruit_scene([(240, 240, 40)])
        regions, _ = fm.find_fruit_regions(rgb)
        self.assertEqual(regions, [])

    def test_small_region_above_current_area_cutoff_is_detected(self):
        rgb = fruit_scene([(240, 240, 50)])
        regions, mode = fm.find_fruit_regions(rgb)
        self.assertEqual(mode, "plain-background")
        self.assertEqual(len(regions), 1)

    def test_touching_fruits_can_merge_into_one_candidate_region(self):
        rgb = fruit_scene([(190, 240, 55), (290, 240, 55)])
        regions, _ = fm.find_fruit_regions(rgb)
        self.assertEqual(len(regions), 1)
        x, _, width, _ = regions[0]
        self.assertLess(x, 190)
        self.assertGreater(width, 150)

    def test_region_count_is_limited_to_current_eight_fruit_cap(self):
        rgb = np.full((600, 600, 3), 255, dtype=np.uint8)
        for y in (110, 300, 490):
            for x in (110, 300, 490):
                cv2.circle(rgb, (x, y), 55, (210, 65, 35), thickness=-1)

        regions, _ = fm.find_fruit_regions(rgb)
        self.assertEqual(len(regions), fm.MAX_FRUITS)

    def test_busy_background_falls_back_to_whole_photo_analysis(self):
        yy, xx = np.indices((480, 480))
        rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=2).astype(np.uint8)
        regions, mode = fm.find_fruit_regions(rgb)
        self.assertEqual(regions, [])
        self.assertEqual(mode, "complex-background")

        with patch.object(fm, "_read_image", return_value=rgb), \
             patch.object(fm, "lighting_report", return_value={}), \
             patch.object(fm, "_try_load_model", return_value=None), \
             patch.object(fm, "_heuristic_freshness", return_value=(55.0, "Moderate", "*", {})), \
             patch.object(fm, "_annotate", return_value="annotated"):
            result = fm.analyze_image(b"synthetic image")

        self.assertEqual(result["mode"], "fallback_heuristic")
        self.assertEqual(result["fruit_count"], 1)
        self.assertEqual(result["fruits"][0]["bbox"], [0, 0, 480, 480])


if __name__ == "__main__":
    unittest.main()
