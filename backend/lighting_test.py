"""
Lighting robustness test for FreshSense.

Takes ONE photo of a fruit, creates versions of it under different lighting
(dim, very dark, bright, overexposed, flat, warm/yellow light, cool/blue light,
half in shadow, glare spot), runs every version through the model and shows
whether the answer stays the same.

Run it from the backend folder:

    python lighting_test.py path\\to\\banana.jpg
    python lighting_test.py path\\to\\banana.jpg --csv results.csv
    python lighting_test.py path\\to\\banana.jpg --save-images lighting_variants

It prints two tables: one with the raw photo, and one with lighting
normalisation turned on, so you can see whether normalisation helps.
"""

import argparse
import csv
import io
import os
import sys

import numpy as np
from PIL import Image

from freshness_model import analyze_image


def _clip(a):
    return np.clip(a, 0, 255).astype("uint8")


def make_variants(rgb):
    """Return {name: rgb_array} - the same photo under different lighting."""
    f = rgb.astype("float32")
    h, w = f.shape[:2]
    mean = f.mean()

    variants = {"original": rgb}
    variants["dim (60%)"] = _clip(f * 0.6)
    variants["very dark (30%)"] = _clip(f * 0.3)
    variants["bright (140%)"] = _clip(f * 1.4)
    variants["overexposed (190%)"] = _clip(f * 1.9)
    variants["flat / low contrast"] = _clip(mean + (f - mean) * 0.4)
    variants["warm yellow light"] = _clip(f * np.array([1.15, 1.05, 0.70]))
    variants["cool blue light"] = _clip(f * np.array([0.80, 0.95, 1.25]))

    half = f.copy()
    half[:, : w // 2] *= 0.35                       # left half in shadow
    variants["half in shadow"] = _clip(half)

    glare = f.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - w * 0.65) ** 2 + (yy - h * 0.3) ** 2) / (2 * (min(h, w) * 0.12) ** 2)))
    glare += blob[..., None] * 255                   # bright spot like a reflection
    variants["glare spot"] = _clip(glare)
    return variants


def to_jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def run(variants, normalize):
    rows = []
    for name, arr in variants.items():
        r = analyze_image(to_jpeg_bytes(arr), normalize=normalize)
        # Compare the predicted fruit type and derived score band across lighting variants.
        signature = (r["is_fruit"], tuple(sorted((f["food"], f["status"]) for f in r["fruits"])) if r["is_fruit"] else ())
        rows.append({
            "variant": name,
            "fruit": r["food"],
            "score": "-" if r["freshness_score"] is None else r["freshness_score"],
            "status": r["status"],
            "mode": "model" if r["mode"] == "trained_model" else "fallback",
            "fruit_model_score": "-" if r["fruit_model_score"] is None else f"{r['fruit_model_score']:.2f}",
            "lighting": r["lighting"]["quality"],
            "warnings": "; ".join(r["lighting"]["warnings"]),
            "_signature": signature,
        })
    base_signature = rows[0]["_signature"]
    for row in rows:
        row["same_as_original"] = row["_signature"] == base_signature
        del row["_signature"]
    return rows


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'variant':<22}{'fruit':<13}{'score':>6}  {'status':<20}{'mode':<9}{'fruit score':>11}  {'same?':<6}lighting")
    for r in rows:
        print(f"{r['variant']:<22}{r['fruit']:<13}{r['score']:>6}  {r['status']:<20}{r['mode']:<9}"
              f"{r['fruit_model_score']:>11}  {'yes' if r['same_as_original'] else 'NO':<6}{r['lighting']}")
    stable = sum(r["same_as_original"] for r in rows)
    print(f"-> Same answer (fruit and score band) as the original in {stable}/{len(rows)} lighting conditions "
          f"({stable / len(rows):.0%} stable)")
    return stable / len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to a photo of a fruit")
    p.add_argument("--csv", help="Save results to this CSV file")
    p.add_argument("--save-images", help="Folder to save the lighting variants as images")
    args = p.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"File not found: {args.image}")

    rgb = np.array(Image.open(args.image).convert("RGB"))
    variants = make_variants(rgb)

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        for name, arr in variants.items():
            safe = "".join(c if c.isalnum() else "_" for c in name)
            Image.fromarray(arr).save(os.path.join(args.save_images, f"{safe}.jpg"))
        print(f"Saved {len(variants)} lighting variants to {args.save_images}")

    raw = run(variants, normalize=False)
    norm = run(variants, normalize=True)
    s_raw = print_table("RAW photo (no lighting correction)", raw)
    s_norm = print_table("WITH lighting normalisation", norm)

    print("\nSummary:")
    if s_norm > s_raw:
        print(f"  Normalisation made the result more stable ({s_raw:.0%} -> {s_norm:.0%}).")
    elif s_norm < s_raw:
        print(f"  Normalisation made it LESS stable ({s_raw:.0%} -> {s_norm:.0%}) - keep it off for this photo.")
    else:
        print(f"  Normalisation made no difference ({s_raw:.0%}).")
    print("  Try the same test on examples from both trained condition labels before deciding.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            cols = ["correction"] + list(raw[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for label, rows in (("none", raw), ("normalised", norm)):
                for r in rows:
                    w.writerow({"correction": label, **r})
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    main()
