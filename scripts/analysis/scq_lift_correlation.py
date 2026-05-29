"""
scq_lift_correlation.py
-----------------------
Compute how well Shot Contest Quality (SCQ) correlates with defensive lift
(actual make rate minus ridge-logit baseline expected make rate), at both
the per-defender level and the per-shot level.

Outputs
-------
  data/outputs/scq_lift_correlation_summary.csv   – per-level correlation table
  data/outputs/scq_lift_defender_detail.csv       – per-defender SCQ vs lift
  stdout                                           – formatted summary
"""

import csv
import math
import argparse
import os
from pathlib import Path


# ── stats helpers (no numpy / scipy required) ──────────────────────────────

def _ranks(vals):
    """Return fractional ranks (1-based) handling ties by average."""
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    n = len(vals)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    return pearson(rx, ry)


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


# ── I/O ────────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots-csv",    default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--defender-csv", default="data/outputs/defender_effectiveness_lift.csv")
    ap.add_argument("--shot-lvl-csv", default="data/outputs/defensive_effectiveness_shot_level.csv")
    ap.add_argument("--out-dir",      default="data/outputs")
    ap.add_argument("--min-shots",    type=int, default=5)
    args = ap.parse_args()

    shots_rows    = read_csv(args.shots_csv)
    defender_rows = read_csv(args.defender_csv)
    shot_lvl_rows = read_csv(args.shot_lvl_csv)

    # ── 1. Per-defender correlation ──────────────────────────────────────
    # Build mean SCQ per defender from shot dataset (analysis-eligible only)
    scq_by_def = {}
    for r in shots_rows:
        if r.get("analysis_eligible", "").strip().lower() not in ("1", "true", "yes"):
            continue
        did = r.get("nearest_defender_id", "").strip()
        dname = r.get("nearest_defender_name", "").strip()
        scq_val = r.get("shot_contest_quality", "").strip()
        if not did or not scq_val:
            continue
        try:
            scq_f = float(scq_val)
        except ValueError:
            continue
        if did not in scq_by_def:
            scq_by_def[did] = {"name": dname, "scqs": [], "count": 0}
        scq_by_def[did]["scqs"].append(scq_f)
        scq_by_def[did]["count"] += 1

    # Build lift per defender from defender_effectiveness_lift.csv
    lift_by_def = {}
    for r in defender_rows:
        did = r.get("nearest_defender_id", "").strip()
        shots = int(r.get("shots", 0))
        if shots < args.min_shots:
            continue
        try:
            lift = float(r["lift_vs_baseline_rate"])
        except (KeyError, ValueError):
            continue
        lift_by_def[did] = lift

    # Join
    defender_detail = []
    for did, info in scq_by_def.items():
        if info["count"] < args.min_shots:
            continue
        if did not in lift_by_def:
            continue
        mean_scq = sum(info["scqs"]) / len(info["scqs"])
        defender_detail.append({
            "defender_id":   did,
            "defender_name": info["name"],
            "shots":         info["count"],
            "mean_scq":      round(mean_scq, 3),
            "lift_vs_baseline_rate": round(lift_by_def[did], 6),
        })

    defender_detail.sort(key=lambda x: x["mean_scq"], reverse=True)

    if len(defender_detail) >= 3:
        xs = [d["mean_scq"] for d in defender_detail]
        ys = [d["lift_vs_baseline_rate"] for d in defender_detail]
        spear_def = round(spearman(xs, ys), 4)
        pear_def  = round(pearson(xs, ys), 4)
        n_def     = len(defender_detail)
    else:
        spear_def = pear_def = float("nan")
        n_def = len(defender_detail)

    # ── 2. Shot-level correlation ─────────────────────────────────────────
    # Join shot_level CSV (has lift_vs_baseline per shot) to shots CSV (has SCQ)
    # Key: (game_id, shooter_id, nearest_defender_id) — use shot index order
    # Simplest: build lookup from shots CSV by row characteristics
    # shot_level rows have game_id, shooter_id, nearest_defender_id, made, p_base, lift_vs_baseline
    # shots CSV rows have game_id, shooter_id, nearest_defender_id, shot_contest_quality

    # Build a dict keyed by (game_id, shooter_id, defender_id) → list of SCQ
    # (multiple shots per pair in same game is possible but rare)
    from collections import defaultdict
    scq_lookup = defaultdict(list)
    for r in shots_rows:
        if r.get("analysis_eligible", "").strip().lower() not in ("1", "true", "yes"):
            continue
        key = (
            r.get("game_id", "").strip(),
            r.get("shooter_id", "").strip(),
            r.get("nearest_defender_id", "").strip(),
        )
        scq_val = r.get("shot_contest_quality", "").strip()
        if scq_val:
            try:
                scq_lookup[key].append(float(scq_val))
            except ValueError:
                pass

    # Match shot-level rows
    shot_pairs = []
    used_indices = defaultdict(int)  # how many times we've pulled from each key
    for r in shot_lvl_rows:
        key = (
            r.get("game_id", "").strip(),
            str(r.get("shooter_id", "")).strip(),
            str(r.get("nearest_defender_id", "")).strip(),
        )
        lift_val = r.get("lift_vs_baseline", "").strip()
        if not lift_val:
            continue
        try:
            lift_f = float(lift_val)
        except ValueError:
            continue
        bucket = scq_lookup.get(key, [])
        idx = used_indices[key]
        if idx >= len(bucket):
            continue
        scq_f = bucket[idx]
        used_indices[key] += 1
        shot_pairs.append((scq_f, lift_f))

    if len(shot_pairs) >= 3:
        xs_s = [p[0] for p in shot_pairs]
        ys_s = [p[1] for p in shot_pairs]
        spear_shot = round(spearman(xs_s, ys_s), 4)
        pear_shot  = round(pearson(xs_s, ys_s), 4)
        n_shot = len(shot_pairs)
    else:
        spear_shot = pear_shot = float("nan")
        n_shot = len(shot_pairs)

    # ── 3. Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCQ ↔ Lift-vs-Baseline Correlation")
    print("=" * 60)

    print(f"\n[Per-defender level]  n = {n_def} defenders (≥{args.min_shots} shots)")
    print(f"  Spearman ρ  : {spear_def:+.4f}")
    print(f"  Pearson  r  : {pear_def:+.4f}")

    print(f"\n[Per-shot level]      n = {n_shot} shots")
    print(f"  Spearman ρ  : {spear_shot:+.4f}")
    print(f"  Pearson  r  : {pear_shot:+.4f}")

    print("\nPer-defender detail (sorted by mean SCQ ↓):")
    print(f"  {'Defender':<25} {'Shots':>5}  {'Mean SCQ':>8}  {'Lift':>8}")
    print("  " + "-" * 52)
    for d in defender_detail:
        lift_str = f"{d['lift_vs_baseline_rate']:+.4f}"
        print(f"  {d['defender_name']:<25} {d['shots']:>5}  {d['mean_scq']:>8.1f}  {lift_str:>8}")

    # interpretation
    print("\nInterpretation:")
    if not math.isnan(spear_def):
        direction = "negative" if spear_def < 0 else "positive"
        strength = "strong" if abs(spear_def) > 0.6 else "moderate" if abs(spear_def) > 0.3 else "weak"
        print(f"  Per-defender: {strength} {direction} relationship (ρ={spear_def:+.3f}).")
        if spear_def < -0.3:
            print("  → Higher SCQ defenders suppress opponent make rate (expected).")
        elif abs(spear_def) < 0.3:
            print("  → SCQ rankings diverge from lift rankings — volume/matchup effects likely.")
    if not math.isnan(spear_shot):
        direction = "negative" if spear_shot < 0 else "positive"
        strength = "strong" if abs(spear_shot) > 0.4 else "moderate" if abs(spear_shot) > 0.2 else "weak"
        print(f"  Per-shot: {strength} {direction} relationship (ρ={spear_shot:+.3f}).")
        if spear_shot < -0.15:
            print("  → Higher-SCQ contests reduce individual shot make probability.")

    # ── 4. Write outputs ──────────────────────────────────────────────────
    summary_rows = [
        {"level": "per_defender", "n": n_def,  "spearman_rho": spear_def,  "pearson_r": pear_def},
        {"level": "per_shot",     "n": n_shot, "spearman_rho": spear_shot, "pearson_r": pear_shot},
    ]
    write_csv(
        os.path.join(args.out_dir, "scq_lift_correlation_summary.csv"),
        summary_rows,
        ["level", "n", "spearman_rho", "pearson_r"],
    )
    write_csv(
        os.path.join(args.out_dir, "scq_lift_defender_detail.csv"),
        defender_detail,
        ["defender_id", "defender_name", "shots", "mean_scq", "lift_vs_baseline_rate"],
    )
    print(f"\nWrote: {args.out_dir}/scq_lift_correlation_summary.csv")
    print(f"Wrote: {args.out_dir}/scq_lift_defender_detail.csv\n")


if __name__ == "__main__":
    main()
