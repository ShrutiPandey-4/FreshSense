"""
"How soon will it spoil?" - a rough shelf-life estimate for each fruit.

How it works
  1. Every fruit has a TYPICAL shelf life when it is fully fresh (table below).
  2. The freshness score says how much of that life is left. The score line for
     "spoiled" is 40, so:   fraction left = (score - 40) / 60   (0 at 40, 1 at 100).
  3. days left = typical shelf life x fraction left.  A range is shown
     (60% to 100% of that number) because nobody can say it exactly.

Honest limits (shown to the user too)
  * It is a rough guide, not a measurement. Real shelf life depends on how ripe the
    fruit was when bought, the temperature, bruises and how it was handled.
  * The trained model's scores are often close to 0 or 100, so the estimate is
    often "plenty of time" or "already spoiled" and rarely something in between.
  * Fruit that the model scores below 40 is reported as spoiled: do not eat it.

Where the typical numbers come from (all approximate; "est." = my own conservative guess)
  apple        room 7   fridge 28   MSU Extension (counter: no longer than 7 days); Texas A&M (ripe apples ~1 month in the fridge)
  banana       room 3   fridge 5    Texas A&M (2-3 days at room temperature); fridge value est.
  grape        room 2   fridge 5    Texas A&M (3-5 days in the fridge); room value est.
  orange       room 14  fridge 28   Texas A&M (about 2 weeks at room temperature); fridge value est.
  strawberry   room 1   fridge 3    Texas A&M (2-3 days in the fridge); room value est.
  guava        room 2   fridge 5    consumer food-storage article (1-2 days / 4-5 days); low-quality source
  pomegranate  room 7   fridge 28   UC Davis Postharvest (up to 2 months at 5 C - shortened here for home fridges); room value est.
  jujube       room 3   fridge 14   a cold-storage study (up to 20 days at 5 C - shortened here); room value est.
Change the numbers in SHELF_LIFE if you find better sources.
"""

import math

SPOILED_BELOW = 40.0          # same line the app uses for "Spoiled / Not Fresh"
RANGE_LOW_FACTOR = 0.6        # the low end of the range is 60% of the estimate

SHELF_LIFE = {
    "apple": {
        "best": "fridge",
        "room": 7, "fridge": 28,
        "tip": "Keep apples in the fridge to make them last several times longer. "
               "Store them away from other fruit - apples give off a gas that ripens neighbours faster.",
    },
    "banana": {
        "best": "room",
        "room": 3, "fridge": 5,
        "tip": "Keep bananas on the counter. In the fridge the peel turns brown but the fruit inside stays "
               "good a little longer. Lots of brown spots means it is very ripe - good for smoothies or baking.",
    },
    "grape": {
        "best": "fridge",
        "room": 2, "fridge": 5,
        "tip": "Keep grapes in the fridge, unwashed, and wash them just before eating. "
               "Remove any shrivelled or mouldy grape - mould spreads to the others.",
    },
    "orange": {
        "best": "room",
        "room": 14, "fridge": 28,
        "tip": "Oranges last about two weeks on the counter and longer in the fridge. "
               "Throw it away if it has soft spots or mould.",
    },
    "strawberry": {
        "best": "fridge",
        "room": 1, "fridge": 3,
        "tip": "Keep strawberries in the fridge, unwashed. One mouldy berry spreads to the rest, "
               "so take it out quickly.",
    },
    "guava": {
        "best": "fridge",
        "room": 2, "fridge": 5,
        "tip": "Ripe guava goes soft quickly. Keep it in the fridge to slow that down.",
    },
    "pomegranate": {
        "best": "fridge",
        "room": 7, "fridge": 28,
        "tip": "Whole pomegranates keep well, especially in the fridge. "
               "Once the seeds are out, eat them within a few days.",
    },
    "jujube": {
        "best": "fridge",
        "room": 3, "fridge": 14,
        "tip": "Fresh jujube turns wrinkly and soft as it ages. Keep it in the fridge to make it last longer.",
    },
}

DISCLAIMER = ("Rough estimate: the typical shelf life of this fruit, scaled by its freshness score. "
              "Real life depends on how ripe it was when bought, the temperature and handling. "
              "The AI's scores are often close to 0 or 100, so use this only as a guide. "
              "If in doubt, do not eat it.")


def _round(x):
    return int(math.floor(x + 0.5))


def _span(days):
    """Turn a number of days into words like '2-3 days' or 'about 3 weeks'."""
    if days < 1:
        return "less than a day"
    lo, hi = _round(days * RANGE_LOW_FACTOR), _round(days)
    if hi >= 14:
        lo_w, hi_w = _round(lo / 7.0), _round(hi / 7.0)
        return f"{hi_w} weeks" if lo_w >= hi_w else f"{max(lo_w, 1)}-{hi_w} weeks"
    if lo >= hi:
        return f"{hi} day" + ("" if hi == 1 else "s")
    return f"{max(lo, 1)}-{hi} days"


def estimate(food, score, uncertain=False):
    """
    Shelf-life estimate for one fruit, or None if we have no data for this fruit.

    food       fruit name, e.g. "Banana"
    score      freshness score 0-100
    uncertain  True if the fruit type or the result itself is doubtful
    """
    info = SHELF_LIFE.get(str(food).strip().lower())
    if info is None or score is None:
        return None

    out = {
        "basis": "typical shelf life of this fruit scaled by the freshness score (rough estimate)",
        "typical_days_room_temperature": info["room"],
        "typical_days_fridge": info["fridge"],
        "storage_tip": info["tip"],
        "disclaimer": DISCLAIMER,
    }

    if score < SPOILED_BELOW:
        out.update(urgency="discard", urgency_label="Already spoiled",
                   days_left_room_temperature=[0, 0], days_left_fridge=None, storage_times="",
                   freshness_life="0 days (spoiled)",
                   message="This fruit already looks spoiled. Do not eat it - throw it away.",
                   storage_tip="Throw it away so it does not spoil the other fruit.")
        return out

    fraction = min(1.0, (float(score) - SPOILED_BELOW) / (100.0 - SPOILED_BELOW))
    room = info["room"] * fraction
    fridge = info["fridge"] * fraction

    best = info["best"]
    best_days = room if best == "room" else fridge
    if best_days < 1:
        urgency, label = "eat_today", "Eat today"
    elif best_days <= 3.5:
        urgency, label = "eat_soon", "Eat within a few days"
    elif best_days < 7:
        urgency, label = "this_week", "Fine for about a week"
    else:
        urgency, label = "plenty_of_time", "Plenty of time"

    if urgency == "plenty_of_time" and score < 70:          # "Moderate" fruit is already ageing
        urgency, label = "this_week", "Fine for about a week"

    place = {"room": "At room temperature", "fridge": "In the fridge"}
    other = "fridge" if best == "room" else "room"
    spans = {"room": _span(room), "fridge": _span(fridge)}
    times = f"{place[best]}: {spans[best]}. {place[other]}: {spans[other]}."
    out.update(
        urgency=urgency, urgency_label=label, best_storage=best, storage_times=times,
        freshness_life=spans["room"],
        days_left_room_temperature=[_round(room * RANGE_LOW_FACTOR), _round(room)],
        days_left_fridge=[_round(fridge * RANGE_LOW_FACTOR), _round(fridge)],
        message=f"{label}. {times}",
    )
    if uncertain:
        out["caution"] = "Result uncertain - rough guide only."
        out["message"] += " " + out["caution"]
    return out
