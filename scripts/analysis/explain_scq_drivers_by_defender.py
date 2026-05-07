#!/usr/bin/env python3
"""
Break down Shot Contest Quality (SCQ) into feature contributions per defender.

SCQ formula from build_unified_shot_dataset.py:
  SCQ = 100 * (
      0.35 * max(0, 1 - contest_distance_ft / 10)
    + 0.30 * clip(closeout_speed_ft_s / 8, 0, 1)
    + 0.20 * max(0, 1 - contest_angle_deg / 90)
    + 0.15 * clip(hand_up_in / 18, 0, 1)
  )

Outputs:
  - defender_scq_driver_breakdown.csv
  - defender_scq_recommendations.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def _to_float(v: str) -> float:
    s = (v or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _scq_components(dist_ft: float, speed_ft_s: float, angle_deg: float, hand_in: float) -> Dict[str, float]:
    d_norm = max(0.0, 1.0 - dist_ft / 10.0)
    s_norm = _clip01(speed_ft_s / 8.0)
    a_norm = max(0.0, 1.0 - angle_deg / 90.0)
    h_norm = _clip01(hand_in / 18.0)
    return {
        "distance_pts": 100.0 * 0.35 * d_norm,
        "speed_pts": 100.0 * 0.30 * s_norm,
        "angle_pts": 100.0 * 0.20 * a_norm,
        "hand_pts": 100.0 * 0.15 * h_norm,
    }


def _recommendations(row: Dict[str, float]) -> Tuple[str, str]:
    deltas = {
        "distance": row["distance_pts_diff_vs_pool"],
        "speed": row["speed_pts_diff_vs_pool"],
        "angle": row["angle_pts_diff_vs_pool"],
        "hand": row["hand_pts_diff_vs_pool"],
    }
    weakest = sorted(deltas.items(), key=lambda kv: kv[1])[:2]
    weak_keys = [k for k, _ in weakest]

    drills: List[str] = []
    film: List[str] = []

    if "distance" in weak_keys:
        drills.append("start 1 step higher on shooter to reduce release-gap distance")
        film.append("review late-arrival reps where contest starts too deep")
    if "speed" in weak_keys:
        drills.append("add first-2-steps acceleration closeout reps")
        film.append("tag low-urgency closeouts and compare to high-speed reps")
    if "angle" in weak_keys:
        drills.append("run angle-control closeouts to stay between shooter and rim")
        film.append("review hips/footwork on fly-by and side-angle contests")
    if "hand" in weak_keys:
        drills.append("emphasize early high-hand timing at shot gather")
        film.append("review hand timing at release frame")

    return "; ".join(drills), "; ".join(film)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain SCQ drivers by defender.")
    parser.add_argument("--csv", default="data/outputs/shot_contest_dataset.csv", help="Input shot dataset.")
    parser.add_argument("--out-dir", default="data/outputs", help="Output directory.")
    parser.add_argument("--min-shots", type=int, default=5, help="Minimum defender shots for report.")
    args = parser.parse_args()

    agg = defaultdict(
        lambda: {
            "shots": 0.0,
            "scq": 0.0,
            "distance_pts": 0.0,
            "speed_pts": 0.0,
            "angle_pts": 0.0,
            "hand_pts": 0.0,
        }
    )
    pool = {"shots": 0.0, "scq": 0.0, "distance_pts": 0.0, "speed_pts": 0.0, "angle_pts": 0.0, "hand_pts": 0.0}

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "nearest_defender_id",
            "nearest_defender_name",
            "contest_distance_ft",
            "closeout_speed_ft_s",
            "contest_angle_deg",
            "hand_up_in",
            "shot_contest_quality",
        ]
        missing = [c for c in required if reader.fieldnames is None or c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            did = (row.get("nearest_defender_id") or "").strip()
            dnm = (row.get("nearest_defender_name") or "").strip()
            if not did or not dnm:
                continue
            dist = _to_float(row.get("contest_distance_ft", ""))
            speed = _to_float(row.get("closeout_speed_ft_s", ""))
            angle = _to_float(row.get("contest_angle_deg", ""))
            hand = _to_float(row.get("hand_up_in", ""))
            scq = _to_float(row.get("shot_contest_quality", ""))
            if not all(math.isfinite(v) for v in [dist, speed, angle, hand, scq]):
                continue

            comps = _scq_components(dist, speed, angle, hand)
            key = (did, dnm)
            agg[key]["shots"] += 1.0
            agg[key]["scq"] += scq
            for k, v in comps.items():
                agg[key][k] += v

            pool["shots"] += 1.0
            pool["scq"] += scq
            for k, v in comps.items():
                pool[k] += v

    if pool["shots"] <= 0:
        raise ValueError("No valid rows to analyze.")

    pool_means = {k: pool[k] / pool["shots"] for k in ["scq", "distance_pts", "speed_pts", "angle_pts", "hand_pts"]}

    breakdown_rows: List[Dict[str, str]] = []
    rec_rows: List[Dict[str, str]] = []

    sorted_items = sorted(agg.items(), key=lambda kv: kv[1]["shots"], reverse=True)
    for (did, dnm), s in sorted_items:
        shots = int(s["shots"])
        if shots < args.min_shots:
            continue
        means = {k: s[k] / s["shots"] for k in ["scq", "distance_pts", "speed_pts", "angle_pts", "hand_pts"]}
        rowf = {
            "nearest_defender_id": did,
            "nearest_defender_name": dnm,
            "shots": float(shots),
            "mean_scq": means["scq"],
            "distance_pts_mean": means["distance_pts"],
            "speed_pts_mean": means["speed_pts"],
            "angle_pts_mean": means["angle_pts"],
            "hand_pts_mean": means["hand_pts"],
            "distance_pts_diff_vs_pool": means["distance_pts"] - pool_means["distance_pts"],
            "speed_pts_diff_vs_pool": means["speed_pts"] - pool_means["speed_pts"],
            "angle_pts_diff_vs_pool": means["angle_pts"] - pool_means["angle_pts"],
            "hand_pts_diff_vs_pool": means["hand_pts"] - pool_means["hand_pts"],
        }
        breakdown_rows.append({k: (f"{v:.6f}" if isinstance(v, float) else str(v)) for k, v in rowf.items()})

        drill_rec, film_focus = _recommendations(rowf)
        rec_rows.append(
            {
                "nearest_defender_id": did,
                "nearest_defender_name": dnm,
                "shots": str(shots),
                "primary_driver_up": max(
                    ["distance_pts_diff_vs_pool", "speed_pts_diff_vs_pool", "angle_pts_diff_vs_pool", "hand_pts_diff_vs_pool"],
                    key=lambda c: rowf[c],
                ).replace("_pts_diff_vs_pool", ""),
                "primary_driver_down": min(
                    ["distance_pts_diff_vs_pool", "speed_pts_diff_vs_pool", "angle_pts_diff_vs_pool", "hand_pts_diff_vs_pool"],
                    key=lambda c: rowf[c],
                ).replace("_pts_diff_vs_pool", ""),
                "coaching_drill_recommendation": drill_rec,
                "film_focus_recommendation": film_focus,
            }
        )

    out_breakdown = f"{args.out_dir}/defender_scq_driver_breakdown.csv"
    out_recs = f"{args.out_dir}/defender_scq_recommendations.csv"

    with open(out_breakdown, "w", newline="", encoding="utf-8") as f:
        fields = [
            "nearest_defender_id",
            "nearest_defender_name",
            "shots",
            "mean_scq",
            "distance_pts_mean",
            "speed_pts_mean",
            "angle_pts_mean",
            "hand_pts_mean",
            "distance_pts_diff_vs_pool",
            "speed_pts_diff_vs_pool",
            "angle_pts_diff_vs_pool",
            "hand_pts_diff_vs_pool",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(breakdown_rows)

    with open(out_recs, "w", newline="", encoding="utf-8") as f:
        fields = [
            "nearest_defender_id",
            "nearest_defender_name",
            "shots",
            "primary_driver_up",
            "primary_driver_down",
            "coaching_drill_recommendation",
            "film_focus_recommendation",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rec_rows)

    print("Wrote:")
    print(" ", out_breakdown)
    print(" ", out_recs)


if __name__ == "__main__":
    main()
