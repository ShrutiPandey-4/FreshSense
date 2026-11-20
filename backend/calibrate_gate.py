"""
Tune the "is it a fruit?" check with YOUR photos.

FreshSense decides "this is not a fruit" when the model's strongest class score
is below a threshold. The default (-1.0) was chosen from a handful of
non-fruit photos. Real fruit photos from your phone or Google can score
differently, so measure it:

  1. Make a folder with 15-30 photos of the fruits FreshSense supports
     (fresh and rotten, different lighting and backgrounds):      my_fruits/
  2. Make a folder with 15-30 photos of things that are NOT fruit
     (people, bottles, books, tables, vegetables, cars ...):      my_others/
  3. From the backend folder run:

         python calibrate_gate.py --fruit_dir my_fruits --other_dir my_others

It prints every photo's score, picks the threshold that separates the two
folders best, and saves it to model/gate_config.json (the backend reads it
automatically - restart the backend afterwards).

Only have fruit photos?  Use --fruit_dir alone: the threshold is then set so
that about 97% of your fruit photos pass (--target_recall to change).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

import freshness_model as fm

EXTS = ("jpg", "jpeg", "png", "webp", "bmp")


def collect(folder):
    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(folder, "**", f"*.{e}"), recursive=True)
        paths += glob.glob(os.path.join(folder, "**", f"*.{e.upper()}"), recursive=True)
    return sorted(set(paths))


def score_folder(folder):
    paths = collect(folder)
    if not paths:
        sys.exit(f"No images found in {folder}")
    imgs, ok = [], []
    for p in paths:
        try:
            imgs.append(np.array(Image.open(p).convert("RGB")))
            ok.append(p)
        except Exception as e:
            print(f"  skipped {p}: {e}")
    scores, kind = [], None
    for i in range(0, len(imgs), 32):                       # batches of 32
        _, g, kind = fm._run_model(imgs[i:i + 32])
        scores += [float(v) for v in g]
    return ok, np.array(scores), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit_dir", required=True, help="folder of photos of the supported fruits")
    ap.add_argument("--other_dir", help="folder of photos that are NOT fruit (recommended)")
    ap.add_argument("--target_recall", type=float, default=0.97,
                    help="share of fruit photos that must pass (default 0.97)")
    ap.add_argument("--no_save", action="store_true", help="only print, do not write gate_config.json")
    args = ap.parse_args()

    if fm._try_load_model() is None:
        sys.exit("No trained model found in the model folder - nothing to calibrate.")

    f_paths, f_scores, kind = score_folder(args.fruit_dir)
    key = "min_max_logit" if kind == "max_logit" else "min_max_softmax"
    print(f"\nScore used: {kind}   (higher = more fruit-like)")
    print(f"\nFRUIT photos ({len(f_scores)}):  lowest {f_scores.min():.2f}   median {np.median(f_scores):.2f}   highest {f_scores.max():.2f}")

    if args.other_dir:
        o_paths, o_scores, _ = score_folder(args.other_dir)
        print(f"OTHER photos ({len(o_scores)}):  lowest {o_scores.min():.2f}   median {np.median(o_scores):.2f}   highest {o_scores.max():.2f}")

        cands = np.unique(np.concatenate([f_scores, o_scores]))
        cands = np.concatenate([[cands.min() - 1], (cands[:-1] + cands[1:]) / 2, [cands.max() + 1]])
        best = None
        for t in cands:
            recall = float((f_scores >= t).mean())              # fruit photos correctly accepted
            reject = float((o_scores < t).mean())               # non-fruit photos correctly rejected
            bal = (recall + reject) / 2
            ok = recall >= args.target_recall
            rank = (ok, bal, -abs(t))                           # prefer: meets recall target, then best balance
            if best is None or rank > best[0]:
                best = (rank, float(t), recall, reject)
        _, thr, recall, reject = best
        print(f"\nBest threshold: {thr:.2f}")
        print(f"  fruit photos accepted        : {recall:.0%}")
        print(f"  non-fruit photos rejected    : {reject:.0%}")
        if recall < args.target_recall:
            print(f"  (could not reach {args.target_recall:.0%} fruit acceptance without letting many non-fruit photos through)")
        print("\nNon-fruit photos that would still be accepted as fruit:")
        wrong = [(os.path.basename(p), s) for p, s in zip(o_paths, o_scores) if s >= thr]
        print("  " + ("\n  ".join(f"{n}  (score {s:.2f})" for n, s in wrong) if wrong else "none"))
        print("\nFruit photos that would be rejected as 'not a fruit':")
        wrong = [(os.path.basename(p), s) for p, s in zip(f_paths, f_scores) if s < thr]
        print("  " + ("\n  ".join(f"{n}  (score {s:.2f})" for n, s in wrong) if wrong else "none"))
    else:
        thr = float(np.percentile(f_scores, (1 - args.target_recall) * 100)) - 0.25
        print(f"\nThreshold so that ~{args.target_recall:.0%} of your fruit photos pass: {thr:.2f}")
        print("(Add --other_dir with non-fruit photos to also check how many of those get rejected.)")

    if not args.no_save:
        cfg = fm._load_gate_config()
        cfg[key] = round(thr, 3)
        cfg["calibrated_on"] = {"fruit_photos": int(len(f_scores)),
                                "other_photos": int(len(o_scores)) if args.other_dir else 0}
        os.makedirs(os.path.dirname(fm.GATE_CONFIG_PATH), exist_ok=True)
        with open(fm.GATE_CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nSaved to {fm.GATE_CONFIG_PATH}\nRestart the backend (python app.py) to use it.")


if __name__ == "__main__":
    main()
