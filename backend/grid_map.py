"""
Grid map - shows WHICH PART of a fruit looks rotten.

How it works
  1. The fruit's box is cut into a grid of squares (default 6 x 6).
  2. Squares that are mostly background are skipped (background = the colour of the box border).
  3. Every remaining square is enlarged and sent to the SAME trained model that judges the whole fruit.
     The model's "rotten" share for that fruit type (rotten / (fresh + rotten)) is the square's rot score.
  4. Squares are coloured:  red = looks rotten, yellow = unsure, green = looks fresh.
  5. The picture, the squares and a one-line summary are returned for the website.

Honest limits
  * The model was trained on whole fruits, not on small pieces. A single square can be judged wrongly,
    so the grid is a rough guide to WHERE the problem is, not an exact measurement.
  * A bigger grid is more detailed but each square holds less information, so it is less reliable.
"""

import base64
import os

import cv2
import numpy as np

DEFAULT_GRID = 6
MIN_GRID, MAX_GRID = 3, 10
ROTTEN_AT = 0.65          # square rot score at or above this = rotten
FRESH_AT = 0.35           # at or below this = fresh; in between = unsure
MIN_FRUIT_SHARE = 0.5     # a square needs at least this much fruit (not background) to be judged
OVERLAP = 0.10            # each square is cut a little larger so edges are not lost
OUT_WIDTH = 420           # width of the picture sent to the website
RED, YELLOW, GREEN = (214, 48, 49), (240, 180, 0), (46, 160, 67)


def grid_size_from_env():
    try:
        n = int(os.environ.get("FRESHSENSE_GRID", DEFAULT_GRID))
    except ValueError:
        n = DEFAULT_GRID
    return max(MIN_GRID, min(MAX_GRID, n))


def _colour_mask(crop):
    """Simple background guess: pixels far from the colour of the border are fruit."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, int(round(min(h, w) * 0.06)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    mask = (np.linalg.norm(lab - bg, axis=2) > 22).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fruit_mask(crop):
    """
    True where the pixel is fruit, False where it is background.
    Uses GrabCut (OpenCV foreground cut-out) started from a rectangle just inside the box;
    falls back to the border-colour guess if GrabCut gives a silly answer.
    """
    h, w = crop.shape[:2]
    mask = None
    try:
        s = min(1.0, 220.0 / max(h, w))
        small = cv2.resize(crop, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        m = np.zeros((sh, sw), np.uint8)
        rect = (int(sw * 0.04), int(sh * 0.04), int(sw * 0.92), int(sh * 0.92))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), m, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
        if 0.10 <= fg.mean() <= 0.97:
            mask = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        mask = None
    if mask is None:
        mask = _colour_mask(crop)
    if mask.mean() < 0.15:           # nothing found (tight box): treat everything as fruit
        mask = np.ones_like(mask)
    return mask.astype(bool)


def _label(rot):
    return "rotten" if rot >= ROTTEN_AT else ("fresh" if rot <= FRESH_AT else "unsure")


def build_grid(crop, food, run_model, interpret, grid=None, mode="absolute"):
    """
    crop       RGB picture of one fruit
    food       fruit name from the detector / classifier
    run_model  function(list of images) -> (probs, gate, kind)
    interpret  function(prob_row, expected_food) -> dict with "rotten_model_score"
    mode       "absolute": a square is rotten if its rot score is high (use when the whole fruit is spoiled).
               "relative": a square is marked only if it is clearly WORSE than the other squares of the same
               fruit (use when the whole fruit is not spoiled). A plain red apple whose squares all look
               alike then shows no rotten spot, instead of every square being marked.
    """
    grid = grid or grid_size_from_env()
    h, w = crop.shape[:2]
    if h < grid * 4 or w < grid * 4:
        return None                                   # too small to split
    mask = fruit_mask(crop)
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)

    cells, patches = [], []
    for r in range(grid):
        for c in range(grid):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            share = float(mask[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
            cell = {"row": r, "col": c, "fruit_share": round(share, 2), "rot_score": None, "label": "background"}
            if share >= MIN_FRUIT_SHARE:
                py, px = int((y1 - y0) * OVERLAP), int((x1 - x0) * OVERLAP)
                patches.append(crop[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)])
                cell["_patch"] = len(patches) - 1
            cells.append(cell)

    if patches:
        probs, _, _ = run_model(patches)
        for cell in cells:
            i = cell.pop("_patch", None)
            if i is not None:
                rot = float(interpret(probs[i], expected_food=food)["rotten_model_score"])
                cell["rot_score"] = round(rot, 3)
                cell["label"] = _label(rot)

    judged = [c for c in cells if c["rot_score"] is not None]
    if mode == "relative" and judged:
        med = float(np.median([c["rot_score"] for c in judged]))
        for c in judged:
            r = c["rot_score"]
            c["label"] = ("rotten" if r >= max(ROTTEN_AT, med + 0.35)
                          else "unsure" if r >= max(0.5, med + 0.2) else "ok")
    n_rot = sum(c["label"] == "rotten" for c in judged)
    n_unsure = sum(c["label"] == "unsure" for c in judged)
    pct = int(round(100.0 * n_rot / len(judged))) if judged else 0
    if not judged:
        summary = "Could not split this fruit into squares."
    elif n_rot == 0 and mode == "relative":
        summary = "No clearly rotten spot found - all squares look alike."
    elif n_rot == 0:
        summary = f"No rotten-looking squares found ({len(judged)} squares checked)."
    else:
        summary = f"{n_rot} of {len(judged)} squares look rotten (about {pct}% of the fruit)."

    return {"grid": grid, "cells": cells, "squares_checked": len(judged), "rotten_squares": n_rot,
            "unsure_squares": n_unsure, "mode": mode, "rotten_percent": pct, "summary": summary,
            "image": _draw(crop, cells, ys, xs),
            "note": "Rough guide: the model was trained on whole fruits, so single squares can be wrong."}


def _draw(crop, cells, ys, xs):
    out = crop.copy()
    over = crop.copy()
    colours = {"rotten": RED, "unsure": YELLOW, "fresh": GREEN}
    for c in cells:
        col = colours.get(c["label"])
        if col is None:
            continue
        r, k = c["row"], c["col"]
        cv2.rectangle(over, (xs[k], ys[r]), (xs[k + 1], ys[r + 1]), col, -1)
    out = cv2.addWeighted(over, 0.42, out, 0.58, 0)
    for y in ys:
        cv2.line(out, (0, int(y)), (out.shape[1] - 1, int(y)), (255, 255, 255), 1)
    for x in xs:
        cv2.line(out, (int(x), 0), (int(x), out.shape[0] - 1), (255, 255, 255), 1)
    s = OUT_WIDTH / out.shape[1]
    out = cv2.resize(out, (OUT_WIDTH, max(1, int(out.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
