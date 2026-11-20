"""Tests for the rot grid (which squares of a fruit look rotten)."""

import unittest

import numpy as np

import grid_map as gm


def _interpret_stub(row, expected_food=None):
    return {"rotten_model_score": float(row[0])}


def _make_fruit():
    """White background, a 'fruit' made of a left half (light) and a right half (dark)."""
    img = np.full((120, 120, 3), 255, np.uint8)
    img[10:110, 10:60] = (200, 150, 40)
    img[10:110, 60:110] = (90, 50, 20)
    return img


def _run_by_darkness(patches):
    # dark patch -> rotten (1.0), light patch -> fresh (0.0)
    probs = np.array([[1.0 if p.mean() < 110 else 0.0] for p in patches])
    return probs, probs[:, 0], "stub"


class GridTests(unittest.TestCase):
    def test_background_squares_are_skipped(self):
        g = gm.build_grid(_make_fruit(), "Apple", _run_by_darkness, _interpret_stub, grid=6)
        labels = {(c["row"], c["col"]): c["label"] for c in g["cells"]}
        self.assertEqual(labels[(0, 0)], "background")
        self.assertEqual(labels[(5, 5)], "background")

    def test_rotten_side_is_marked_and_fresh_side_is_not(self):
        g = gm.build_grid(_make_fruit(), "Apple", _run_by_darkness, _interpret_stub, grid=6)
        by = {(c["row"], c["col"]): c["label"] for c in g["cells"]}
        self.assertEqual(by[(3, 1)], "fresh")      # left half
        self.assertEqual(by[(3, 4)], "rotten")     # right half
        self.assertGreater(g["rotten_squares"], 0)
        self.assertIn("squares look rotten", g["summary"])

    def test_all_fresh_fruit_says_no_rotten_squares(self):
        light = np.full((120, 120, 3), 255, np.uint8)
        light[10:110, 10:110] = (200, 150, 40)
        g = gm.build_grid(light, "Apple", _run_by_darkness, _interpret_stub, grid=6)
        self.assertEqual(g["rotten_squares"], 0)
        self.assertTrue(g["summary"].startswith("No rotten-looking squares"))

    def test_square_counts_add_up(self):
        g = gm.build_grid(_make_fruit(), "Apple", _run_by_darkness, _interpret_stub, grid=6)
        judged = [c for c in g["cells"] if c["rot_score"] is not None]
        self.assertEqual(len(judged), g["squares_checked"])
        self.assertEqual(len(g["cells"]), 36)

    def test_tiny_crop_returns_none(self):
        tiny = np.zeros((10, 10, 3), np.uint8)
        self.assertIsNone(gm.build_grid(tiny, "Apple", _run_by_darkness, _interpret_stub, grid=6))

    def test_picture_is_a_png_data_uri(self):
        g = gm.build_grid(_make_fruit(), "Apple", _run_by_darkness, _interpret_stub, grid=6)
        self.assertTrue(g["image"].startswith("data:image/png;base64,"))

    def test_unsure_band(self):
        self.assertEqual(gm._label(0.5), "unsure")
        self.assertEqual(gm._label(0.9), "rotten")
        self.assertEqual(gm._label(0.1), "fresh")

    def test_grid_size_is_limited(self):
        import os
        os.environ["FRESHSENSE_GRID"] = "99"
        try:
            self.assertEqual(gm.grid_size_from_env(), gm.MAX_GRID)
            os.environ["FRESHSENSE_GRID"] = "abc"
            self.assertEqual(gm.grid_size_from_env(), gm.DEFAULT_GRID)
        finally:
            del os.environ["FRESHSENSE_GRID"]


class AttachGridTests(unittest.TestCase):
    def _result(self, status):
        return {"fruits": [{"food": "Apple", "bbox": [0, 0, 120, 120], "status": status}]}

    def _attach(self, status, run):
        import freshness_model as fm
        from unittest.mock import patch
        res = self._result(status)
        with patch.object(fm, "_run_model", run), patch.object(fm, "_interpret", _interpret_stub):
            fm._attach_grids(_make_fruit(), res)
        return res["fruits"][0]["rot_grid"]

    def test_spoiled_fruit_with_no_rotten_squares_gets_a_caution(self):
        g = self._attach("Spoiled / Not Fresh", lambda p: (np.zeros((len(p), 1)), np.zeros(len(p)), "stub"))
        self.assertTrue(g.get("disagrees_with_whole_fruit"))
        self.assertIn("disagree", g["summary"])

    def test_agreeing_result_has_no_caution(self):
        g = self._attach("Spoiled / Not Fresh", _run_by_darkness)
        self.assertNotIn("disagrees_with_whole_fruit", g)

    def test_grid_failure_never_breaks_the_result(self):
        import freshness_model as fm
        from unittest.mock import patch
        res = self._result("Fresh")
        with patch.object(fm.grid_map, "build_grid", side_effect=RuntimeError("boom")):
            fm._attach_grids(_make_fruit(), res)
        self.assertNotIn("rot_grid", res["fruits"][0])


class RelativeModeTests(unittest.TestCase):
    def test_plain_fruit_whose_squares_all_score_high_shows_no_rotten_spot(self):
        all_high = lambda p: (np.full((len(p), 1), 0.95), np.zeros(len(p)), "stub")
        g = gm.build_grid(_make_fruit(), "Apple", all_high, _interpret_stub, grid=6, mode="relative")
        self.assertEqual(g["rotten_squares"], 0)
        self.assertIn("No clearly rotten spot", g["summary"])

    def test_one_bad_patch_is_still_found_in_relative_mode(self):
        g = gm.build_grid(_make_fruit(), "Apple", _run_by_darkness, _interpret_stub, grid=6, mode="relative")
        by = {(c["row"], c["col"]): c["label"] for c in g["cells"]}
        self.assertEqual(by[(3, 1)], "ok")
        self.assertEqual(by[(3, 4)], "rotten")

    def test_absolute_mode_still_marks_everything_when_all_high(self):
        all_high = lambda p: (np.full((len(p), 1), 0.95), np.zeros(len(p)), "stub")
        g = gm.build_grid(_make_fruit(), "Apple", all_high, _interpret_stub, grid=6, mode="absolute")
        self.assertEqual(g["rotten_squares"], g["squares_checked"])


if __name__ == "__main__":
    unittest.main()
