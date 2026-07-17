#!/usr/bin/env python3
"""
Detect 3PT make/miss directly from Hawk-Eye ball tracking (no play-by-play).

Motivation: NBA play-by-play is unreachable from this environment, and we want
to LABEL Miami Heat 3PA (where opponents defend) to roughly double the sample
for the shot-value / pitch-control validation. A made 3 sends the ball cleanly
DOWN through the rim cylinder; a miss does not. We detect the first descending
crossing of rim height near the rim and measure the ball's horizontal distance
to the rim center there. Small -> through the hoop -> MADE.

This script (a) classifies every shot in a CSV from tracking, and (b) if the CSV
carries pbp_shot_result, validates the detector against those ground-truth
labels so we know its accuracy before trusting it on unlabeled Heat shots.

Usage:
  python detect_make_from_tracking.py --validate          # check vs pbp labels
  python detect_make_from_tracking.py --shots-csv X --output-csv Y
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)

RIM_Z = 120.0          # rim height (inches)
SEARCH_AHEAD = 170     # frames after release to look for the rim crossing
SEARCH_BEHIND = 5
NEAR_RIM_IN = 24.0     # only count rim-height crossings within this horiz dist
MADE_THRESHOLD_IN = 11.0  # ball center within this of rim center at crossing -> made


def _to_int(v: str) -> Optional[int]:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_ball_track(path: str) -> Dict[int, Tuple[float, float, float]]:
    t = pq.read_table(path, columns=["frame", "ball_x", "ball_y", "ball_z"])
    fr = t["frame"].to_numpy(zero_copy_only=False)
    bx = t["ball_x"].to_numpy(zero_copy_only=False)
    by = t["ball_y"].to_numpy(zero_copy_only=False)
    bz = t["ball_z"].to_numpy(zero_copy_only=False)
    ball: Dict[int, Tuple[float, float, float]] = {}
    for f, x, y, z in zip(fr, bx, by, bz):
        ball[int(f)] = (float(x), float(y), float(z))
    return ball


def detect_make(ball: Dict[int, Tuple[float, float, float]], rf: int,
                rim_x: float, rim_y: float,
                threshold: float = MADE_THRESHOLD_IN) -> Tuple[int, float]:
    """Return (made 0/1, min horizontal distance to rim at a descending crossing).

    A descending crossing of rim height that happens near the rim is the shot
    reaching the basket. The closest such approach to the rim center decides
    make vs miss. No near-rim crossing at all -> miss (air-balled / long).
    """
    best = float("inf")
    for f in range(rf - SEARCH_BEHIND, rf + SEARCH_AHEAD):
        a = ball.get(f)
        b = ball.get(f + 1)
        if a is None or b is None:
            continue
        # Ball descending through rim height.
        if a[2] >= RIM_Z >= b[2]:
            for ff in range(f - 2, f + 3):
                c = ball.get(ff)
                if c is None:
                    continue
                hd = math.hypot(c[0] - rim_x, c[1] - rim_y)
                if hd < best:
                    best = hd
    if not math.isfinite(best) or best > NEAR_RIM_IN:
        return 0, (best if math.isfinite(best) else 99.0)
    return (1 if best <= threshold else 0), best


def classify_shots(shots: List[dict], input_dir: str) -> List[dict]:
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for s in shots:
        by_game[s["game_file"]].append(s)
    out: List[dict] = []
    for gf in sorted(by_game):
        path = os.path.join(input_dir, gf)
        if not os.path.exists(path):
            print(f"  WARNING: missing {gf}")
            continue
        ball = load_ball_track(path)
        for s in by_game[gf]:
            rf = _to_int(s["release_frame"])
            rim_x = float(s["rim_x"])
            rim_y = float(s["rim_y"])
            made, dist = detect_make(ball, rf, rim_x, rim_y)
            r = dict(s)
            r["tracking_made"] = made
            r["rim_crossing_horiz_in"] = round(dist, 2)
            out.append(r)
        del ball
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots-csv", default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--validate", action="store_true",
                    help="Compare tracking detection against pbp_shot_result.")
    ap.add_argument("--threshold", type=float, default=MADE_THRESHOLD_IN)
    args = ap.parse_args()

    with open(args.shots_csv, newline="", encoding="utf-8") as f:
        shots = list(csv.DictReader(f))

    classified = classify_shots(shots, args.input_dir)

    if args.validate:
        # Sweep threshold to report best accuracy vs pbp ground truth.
        labeled = [s for s in classified
                   if any(k in (s.get("pbp_shot_result", "") or "").lower()
                          for k in ("made", "miss"))]
        y = [1 if "made" in s["pbp_shot_result"].lower() else 0 for s in labeled]
        dist = [s["rim_crossing_horiz_in"] for s in labeled]
        print(f"Validation on {len(labeled)} pbp-labeled shots ({sum(y)} made):")
        best = None
        for thr in [round(t, 1) for t in _frange(5.0, 18.0, 0.5)]:
            pred = [1 if d <= thr else 0 for d in dist]
            acc = sum(int(p == yi) for p, yi in zip(pred, y)) / len(y)
            if best is None or acc > best[1]:
                best = (thr, acc)
        thr = best[0]
        pred = [1 if d <= thr else 0 for d in dist]
        tp = sum(1 for p, yi in zip(pred, y) if p == 1 and yi == 1)
        tn = sum(1 for p, yi in zip(pred, y) if p == 0 and yi == 0)
        fp = sum(1 for p, yi in zip(pred, y) if p == 1 and yi == 0)
        fn = sum(1 for p, yi in zip(pred, y) if p == 0 and yi == 1)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  Best threshold: {thr} in -> accuracy {best[1]:.3f}")
        print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Precision(made)={prec:.3f}  Recall(made)={rec:.3f}")
        print(f"  Detected make rate {sum(pred)/len(pred):.3f} vs actual {sum(y)/len(y):.3f}")

    if args.output_csv:
        fieldnames = list(classified[0].keys())
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(classified)
        print(f"Wrote {len(classified)} rows to {args.output_csv}")


def _frange(a: float, b: float, step: float):
    x = a
    while x <= b + 1e-9:
        yield x
        x += step


if __name__ == "__main__":
    main()
