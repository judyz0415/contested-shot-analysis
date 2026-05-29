#!/usr/bin/env python3
"""
SCQ weight sensitivity analysis.

Tests how stable the per-defender SCQ rankings are under 5 different
weight schemes.  For each scheme we report:
  - mean SCQ per defender (≥ MIN_SHOTS)
  - rank
  - Spearman rank correlation vs the original scheme

Output: data/outputs/scq_weight_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple


NA_TOKENS = {"", "NA", "NAN", "NONE", "NULL"}


def _to_float(v: str) -> float:
    s = (v or "").strip()
    if not s or s.upper() in NA_TOKENS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _scq_weighted(
    dist_ft: float,
    speed_ft_s: float,
    angle_deg: float,
    hand_in: float,
    *,
    w_dist: float,
    w_speed: float,
    w_angle: float,
    w_hand: float,
) -> float:
    d_norm = max(0.0, 1.0 - dist_ft / 10.0)
    s_norm = min(1.0, max(0.0, speed_ft_s / 8.0))
    a_norm = min(1.0, max(0.0, (angle_deg - 90.0) / 90.0))
    h_norm = min(1.0, max(0.0, hand_in / 18.0))
    return 100.0 * (w_dist * d_norm + w_speed * s_norm + w_angle * a_norm + w_hand * h_norm)


def _spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    n = len(a)
    if n < 2:
        return float("nan")

    def _ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(order, 1):
            ranks[idx] = float(rank)
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_b) ** 2 for i in range(n)))
    if den_a < 1e-12 or den_b < 1e-12:
        return float("nan")
    return num / (den_a * den_b)


# ---------------------------------------------------------------------------
# Weight schemes
# ---------------------------------------------------------------------------

SCHEMES: List[Tuple[str, Dict[str, float]]] = [
    (
        "original",
        dict(w_dist=0.35, w_speed=0.30, w_angle=0.20, w_hand=0.15),
    ),
    (
        "equal_weights",
        dict(w_dist=0.25, w_speed=0.25, w_angle=0.25, w_hand=0.25),
    ),
    (
        "no_speed_stationed",
        # Speed dropped entirely; weight redistributed to dist/angle/hand
        # Rationale: stationed contests — the defender is already there,
        # speed reflects position not contest quality
        dict(w_dist=0.45, w_speed=0.00, w_angle=0.30, w_hand=0.25),
    ),
    (
        "technique_heavy",
        # Up-weight positioning (angle) and hand; down-weight raw distance
        dict(w_dist=0.20, w_speed=0.15, w_angle=0.35, w_hand=0.30),
    ),
    (
        "distance_heavy",
        # Proximity is king
        dict(w_dist=0.55, w_speed=0.20, w_angle=0.15, w_hand=0.10),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/intermediate/shot_contest_dataset.csv")
    parser.add_argument("--out-dir", default="data/outputs")
    parser.add_argument("--min-shots", type=int, default=5)
    args = parser.parse_args()

    # ── per-defender sums per scheme ────────────────────────────────────────
    agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"shots": 0.0, **{name: 0.0 for name, _ in SCHEMES}}
    )

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("analysis_eligible") or "").strip().lower() != "yes":
                continue
            did  = (row.get("nearest_defender_id") or "").strip()
            dnm  = (row.get("nearest_defender_name") or "").strip()
            if not did or not dnm:
                continue

            dist  = _to_float(row.get("contest_distance_ft", ""))
            speed = _to_float(row.get("closeout_speed_ft_s", ""))
            angle = _to_float(row.get("contest_angle_deg", ""))
            hand  = _to_float(row.get("hand_up_in", ""))
            if not all(math.isfinite(v) for v in [dist, speed, angle, hand]):
                continue

            key = (did, dnm)
            agg[key]["shots"] += 1.0
            for name, weights in SCHEMES:
                agg[key][name] += _scq_weighted(dist, speed, angle, hand, **weights)

    # ── compute means and rank per scheme ───────────────────────────────────
    defenders = [(k, v) for k, v in agg.items() if int(v["shots"]) >= args.min_shots]
    defenders.sort(key=lambda x: x[1]["shots"], reverse=True)

    scheme_names = [name for name, _ in SCHEMES]
    mean_matrix: Dict[str, List[float]] = {name: [] for name in scheme_names}
    defender_names: List[str] = []

    for (did, dnm), s in defenders:
        n = s["shots"]
        defender_names.append(dnm)
        for name in scheme_names:
            mean_matrix[name].append(s[name] / n)

    # Ranks: 1 = highest SCQ
    rank_matrix: Dict[str, List[int]] = {}
    for name in scheme_names:
        vals = mean_matrix[name]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        ranks = [0] * len(vals)
        for rank, idx in enumerate(order, 1):
            ranks[idx] = rank
        rank_matrix[name] = ranks

    # Spearman vs original
    orig_ranks = rank_matrix["original"]
    spearmans = {}
    for name in scheme_names:
        spearmans[name] = _spearman(
            [float(r) for r in orig_ranks],
            [float(r) for r in rank_matrix[name]],
        )

    # ── print summary ────────────────────────────────────────────────────────
    print(f"\nPool: {sum(int(v['shots']) for _, v in defenders)} eligible shots | "
          f"{len(defenders)} defenders (≥{args.min_shots} shots)\n")

    print("Spearman ρ vs original weights:")
    for name in scheme_names:
        print(f"  {name:30s}: ρ = {spearmans[name]:.3f}")

    print("\nPer-defender SCQ means and ranks:")
    header = f"{'Defender':22s} {'Shots':>5s}  " + \
             "  ".join(f"{name[:12]:>12s}" for name in scheme_names)
    print(header)
    print("-" * len(header))
    for i, dnm in enumerate(defender_names):
        shots = int(defenders[i][1]["shots"])
        row_str = f"{dnm:22s} {shots:>5d}  "
        for name in scheme_names:
            mean = mean_matrix[name][i]
            rank = rank_matrix[name][i]
            row_str += f"  {mean:6.1f}(#{rank:2d})"
        print(row_str)

    # ── max rank swing per defender ──────────────────────────────────────────
    print("\nMax rank swing across all schemes (vs original):")
    for i, dnm in enumerate(defender_names):
        orig_r = rank_matrix["original"][i]
        max_swing = max(abs(rank_matrix[name][i] - orig_r) for name in scheme_names)
        print(f"  {dnm:22s}: original rank #{orig_r:2d}, max swing = {max_swing:+d} spots")

    # ── write CSV ────────────────────────────────────────────────────────────
    out_path = f"{args.out_dir}/scq_weight_sensitivity.csv"
    fields = (
        ["defender_id", "defender_name", "shots"]
        + [f"{name}_mean_scq" for name in scheme_names]
        + [f"{name}_rank" for name in scheme_names]
        + [f"spearman_vs_original_{name}" for name in scheme_names]
    )
    rows_out = []
    for i, ((did, dnm), s) in enumerate(defenders):
        r: Dict[str, str] = {
            "defender_id": did,
            "defender_name": dnm,
            "shots": str(int(s["shots"])),
        }
        for name in scheme_names:
            r[f"{name}_mean_scq"] = f"{mean_matrix[name][i]:.2f}"
            r[f"{name}_rank"] = str(rank_matrix[name][i])
            r[f"spearman_vs_original_{name}"] = f"{spearmans[name]:.4f}"
        rows_out.append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
