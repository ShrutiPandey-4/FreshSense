"""Tests for the score softening: scores are never shown as exactly 0 or 100."""

import unittest
from unittest.mock import patch

import numpy as np

import freshness_model as fm


class ScoreSofteningTests(unittest.TestCase):
    def test_extreme_scores_are_never_shown_as_0_or_100(self):
        with patch.object(fm, "SCORE_SOFTENING", 2.0):
            self.assertEqual(fm.soften_score(1.0), fm.SCORE_CEILING)
            self.assertEqual(fm.soften_score(0.0), fm.SCORE_FLOOR)
            self.assertLess(fm.SCORE_CEILING, 100)
            self.assertGreater(fm.SCORE_FLOOR, 0)

    def test_very_sure_results_spread_out_instead_of_all_showing_100(self):
        with patch.object(fm, "SCORE_SOFTENING", 2.0):
            scores = [round(fm.soften_score(p), 1) for p in (0.99, 0.999, 0.9999)]
        self.assertEqual(len(set(scores)), 3)
        self.assertTrue(all(85 < s <= 98 for s in scores), scores)

    def test_even_split_stays_50_and_order_is_kept(self):
        with patch.object(fm, "SCORE_SOFTENING", 2.0):
            self.assertAlmostEqual(fm.soften_score(0.5), 50.0, places=6)
            values = [fm.soften_score(p / 100) for p in range(0, 101, 5)]
        self.assertEqual(values, sorted(values))

    def test_softening_one_switches_it_off(self):
        with patch.object(fm, "SCORE_SOFTENING", 1.0):
            self.assertEqual(fm.soften_score(1.0), 100.0)
            self.assertEqual(fm.soften_score(0.6), 60.0)

    def test_interpret_reports_both_scores(self):
        labels = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten"]
        with patch.object(fm, "_labels", labels), patch.object(fm, "SCORE_SOFTENING", 2.0):
            result = fm._interpret(np.array([0.9, 0.001, 0.05, 0.049]))
            item = fm._fruit_item(1, (0, 0, 8, 8), result, 0.9)
        self.assertEqual(result["score_before_softening"], 99.9)
        self.assertLess(result["score"], 99.0)
        self.assertEqual(item["details"]["score_before_softening"], 99.9)
        self.assertEqual(item["status"], "Fresh")

    def test_status_follows_the_displayed_score(self):
        labels = ["apple_fresh", "apple_rotten"]
        with patch.object(fm, "_labels", labels), patch.object(fm, "SCORE_SOFTENING", 2.0):
            for fresh, rotten in [(0.95, 0.05), (0.8, 0.2), (0.6, 0.4), (0.3, 0.7), (0.01, 0.99)]:
                result = fm._interpret(np.array([fresh, rotten]))
                item = fm._fruit_item(1, (0, 0, 8, 8), result, 0.9)
                self.assertEqual(item["status"], fm._status_from_score(result["score"])[0])

    def test_bad_environment_value_falls_back_to_default(self):
        with patch.dict("os.environ", {"FRESHSENSE_SCORE_SOFTENING": "abc"}):
            self.assertEqual(fm._softening_from_env(), 2.0)
        with patch.dict("os.environ", {"FRESHSENSE_SCORE_SOFTENING": "0.2"}):
            self.assertEqual(fm._softening_from_env(), 1.0)


if __name__ == "__main__":
    unittest.main()
