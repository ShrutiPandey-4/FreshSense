"""Tests for the "how soon will it spoil?" shelf-life estimate."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import freshness_model as fm
import shelf_life as sl


class ShelfLifeEstimateTests(unittest.TestCase):
    def test_spoiled_score_says_discard_and_gives_no_days(self):
        result = sl.estimate("Banana", 12.0)
        self.assertEqual(result["urgency"], "discard")
        self.assertEqual(result["days_left_room_temperature"], [0, 0])
        self.assertIsNone(result["days_left_fridge"])
        self.assertIn("Do not eat", result["message"])

    def test_score_exactly_at_spoiled_line_is_not_discarded_but_eat_today(self):
        result = sl.estimate("Banana", 40.0)
        self.assertEqual(result["urgency"], "eat_today")

    def test_fully_fresh_banana_gets_about_its_typical_room_life(self):
        result = sl.estimate("Banana", 100.0)
        self.assertEqual(result["days_left_room_temperature"][1], sl.SHELF_LIFE["banana"]["room"])
        self.assertLess(result["days_left_room_temperature"][0], result["days_left_room_temperature"][1])
        self.assertEqual(result["urgency"], "eat_soon")

    def test_lower_score_never_gives_more_days(self):
        scores = [100, 90, 80, 70, 60, 50, 41]
        for food in sl.SHELF_LIFE:
            days = [sl.estimate(food, s)["days_left_room_temperature"][1] for s in scores]
            self.assertEqual(days, sorted(days, reverse=True), food)

    def test_fridge_is_never_shorter_than_room_temperature(self):
        for food in sl.SHELF_LIFE:
            for score in (100, 80, 60):
                r = sl.estimate(food, score)
                self.assertGreaterEqual(r["days_left_fridge"][1], r["days_left_room_temperature"][1], food)

    def test_moderate_fruit_is_never_plenty_of_time(self):
        for food in sl.SHELF_LIFE:
            self.assertNotEqual(sl.estimate(food, 65.0)["urgency"], "plenty_of_time", food)

    def test_unknown_fruit_or_missing_score_returns_none(self):
        self.assertIsNone(sl.estimate("Mango", 90))
        self.assertIsNone(sl.estimate("Unknown", 90))
        self.assertIsNone(sl.estimate("Banana", None))

    def test_fruit_name_is_case_insensitive(self):
        self.assertEqual(sl.estimate("APPLE", 90)["message"], sl.estimate("apple", 90)["message"])

    def test_uncertain_result_adds_a_caution(self):
        self.assertIn("caution", sl.estimate("Apple", 90, uncertain=True))
        self.assertNotIn("caution", sl.estimate("Apple", 90, uncertain=False))

    def test_every_supported_fruit_has_complete_data(self):
        names = {c["fruit"] for c in __import__("json").loads(
            (Path(__file__).parent / "model" / "metadata.json").read_text())["classes"]}
        for name in names:
            info = sl.SHELF_LIFE.get(name)
            self.assertIsNotNone(info, f"no shelf-life data for {name}")
            self.assertIn(info["best"], ("room", "fridge"))
            self.assertGreater(info["room"], 0)
            self.assertGreaterEqual(info["fridge"], info["room"])
            self.assertTrue(info["tip"])

    def test_estimate_always_carries_the_rough_guide_disclaimer(self):
        for score in (100, 70, 30):
            self.assertIn("Rough estimate", sl.estimate("Orange", score)["disclaimer"])

    def test_freshness_life_is_a_short_text_that_follows_the_score(self):
        self.assertEqual(sl.estimate("Banana", 100.0)["freshness_life"], "2-3 days")
        self.assertEqual(sl.estimate("Banana", 70.0)["freshness_life"], "1-2 days")
        self.assertEqual(sl.estimate("Banana", 41.0)["freshness_life"], "less than a day")
        self.assertEqual(sl.estimate("Banana", 10.0)["freshness_life"], "0 days (spoiled)")
        self.assertEqual(sl.estimate("Orange", 100.0)["freshness_life"], "1-2 weeks")

    def test_span_wording(self):
        self.assertEqual(sl._span(0.4), "less than a day")
        self.assertEqual(sl._span(1.0), "1 day")
        self.assertEqual(sl._span(3.0), "2-3 days")
        self.assertEqual(sl._span(28.0), "2-4 weeks")


class ShelfLifeInResultTests(unittest.TestCase):
    def _item(self, food, fresh, rotten, weak=False):
        row = np.zeros(len(fm._labels))
        row[fm._labels.index(f"{food}_fresh")] = fresh
        row[fm._labels.index(f"{food}_rotten")] = rotten
        return fm._fruit_item(1, (0, 0, 10, 10), fm._interpret(row), gate_score=3.0, weak=weak)

    @classmethod
    def setUpClass(cls):
        if fm._try_load_model() is None:
            raise unittest.SkipTest("trained model not available")

    def test_each_fruit_result_carries_its_own_shelf_life(self):
        fresh = self._item("banana", 0.9, 0.05)
        rotten = self._item("banana", 0.02, 0.93)
        self.assertEqual(fresh["shelf_life"]["urgency"], "eat_soon")
        self.assertEqual(rotten["shelf_life"]["urgency"], "discard")

    def test_single_fruit_top_level_result_has_shelf_life(self):
        item = self._item("apple", 0.9, 0.05)
        result = fm._build_result(np.zeros((20, 20, 3), np.uint8), [item], [], "test",
                                  {"quality": "good", "brightness": 0.5, "contrast": 0.2, "warnings": []}, False)
        self.assertEqual(result["shelf_life"], item["shelf_life"])

    def test_heuristic_fallback_has_no_shelf_life_because_fruit_is_unknown(self):
        rgb = np.full((60, 60, 3), 150, np.uint8)
        result = fm._fallback_result(rgb, {"quality": "good", "brightness": 0.5, "contrast": 0.2, "warnings": []},
                                     False, "test")
        self.assertNotIn("shelf_life", result["fruits"][0])


if __name__ == "__main__":
    unittest.main()
