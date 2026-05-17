#!/usr/bin/env python3
"""
Estimate defender effectiveness as lift vs shooter/context baseline.

Modeling setup:
1) Baseline model (no defender identity, no contest terms):
   logit(make) ~ shooter_skill + shot_difficulty_controls
2) Full model (adds contest and defender terms with shrinkage):
   logit(make) ~ baseline_terms + contest_terms + defender_random_effect_proxy

The defender effect is interpreted from residual lift vs baseline:
  lift_vs_baseline = actual_make_rate - baseline_expected_make_rate
not from raw FG% allowed.

This script is stdlib-only and writes analysis artifacts as CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


NA_TOKENS = {"", "NA", "NAN", "NONE", "NULL"}


def _to_float(value: str) -> float:
    s = (value or "").strip()
    if not s or s.upper() in NA_TOKENS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _made_binary(value: str) -> float:
    s = (value or "").strip().lower()
    if not s or s.upper() in NA_TOKENS:
        return float("nan")
    if "made" in s:
        return 1.0
    if "miss" in s:
        return 0.0
    return float("nan")


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class Standardizer:
    means: List[float]
    stds: List[float]

    def transform_row(self, row: Sequence[float]) -> List[float]:
        out: List[float] = []
        for i, v in enumerate(row):
            mu = self.means[i]
            sd = self.stds[i]
            out.append((v - mu) / sd if sd > 1e-12 else 0.0)
        return out

    def transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.transform_row(r) for r in rows]


def _fit_standardizer(rows: Sequence[Sequence[float]]) -> Standardizer:
    p = len(rows[0])
    means = [0.0] * p
    stds = [0.0] * p
    n = float(len(rows))

    for j in range(p):
        means[j] = sum(r[j] for r in rows) / n
    for j in range(p):
        var = sum((r[j] - means[j]) ** 2 for r in rows) / max(n - 1.0, 1.0)
        stds[j] = math.sqrt(var) if var > 0 else 1.0
    return Standardizer(means=means, stds=stds)


def _logloss(y: Sequence[float], p: Sequence[float]) -> float:
    eps = 1e-12
    total = 0.0
    for yi, pi in zip(y, p):
        q = min(max(pi, eps), 1.0 - eps)
        total += -(yi * math.log(q) + (1.0 - yi) * math.log(1.0 - q))
    return total / max(len(y), 1)


def _brier(y: Sequence[float], p: Sequence[float]) -> float:
    return sum((yi - pi) ** 2 for yi, pi in zip(y, p)) / max(len(y), 1)


class RidgeLogit:
    def __init__(
        self,
        lr: float = 0.05,
        max_iter: int = 6000,
        l2: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.seed = seed
        self.w: List[float] = []
        self.b: float = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        n = len(y)
        p = len(x[0])
        rng = random.Random(self.seed)
        self.w = [rng.uniform(-0.01, 0.01) for _ in range(p)]
        self.b = 0.0

        for _ in range(self.max_iter):
            grad_w = [0.0] * p
            grad_b = 0.0
            for i in range(n):
                z = self.b + sum(self.w[j] * x[i][j] for j in range(p))
                pi = _sigmoid(z)
                err = pi - y[i]
                grad_b += err
                for j in range(p):
                    grad_w[j] += err * x[i][j]

            inv_n = 1.0 / n
            grad_b *= inv_n
            for j in range(p):
                # L2 penalty does not apply to intercept.
                grad_w[j] = grad_w[j] * inv_n + self.l2 * self.w[j] * inv_n

            self.b -= self.lr * grad_b
            for j in range(p):
                self.w[j] -= self.lr * grad_w[j]

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        p = len(self.w)
        out: List[float] = []
        for row in x:
            z = self.b + sum(self.w[j] * row[j] for j in range(p))
            out.append(_sigmoid(z))
        return out


def _build_defender_ohe(ids: Sequence[str]) -> Tuple[Dict[str, int], List[List[float]]]:
    uniq = sorted({v for v in ids if v})
    if not uniq:
        return {}, [[0.0] for _ in ids]
    baseline = uniq[0]
    mapping: Dict[str, int] = {}
    k = 0
    for d in uniq:
        if d == baseline:
            continue
        mapping[d] = k
        k += 1
    out: List[List[float]] = []
    for d in ids:
        row = [0.0] * len(mapping)
        if d in mapping:
            row[mapping[d]] = 1.0
        out.append(row)
    return mapping, out


def _calibration_bins(
    y: Sequence[float], p: Sequence[float], n_bins: int = 10
) -> List[Dict[str, str]]:
    pairs = sorted(zip(p, y), key=lambda z: z[0])
    n = len(pairs)
    rows: List[Dict[str, str]] = []
    if n == 0:
        return rows
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = (b + 1) * n // n_bins
        if lo >= hi:
            continue
        chunk = pairs[lo:hi]
        preds = [v[0] for v in chunk]
        obs = [v[1] for v in chunk]
        rows.append(
            {
                "bin": str(b + 1),
                "n": str(len(chunk)),
                "pred_mean": f"{sum(preds)/len(preds):.6f}",
                "obs_rate": f"{sum(obs)/len(obs):.6f}",
                "pred_min": f"{preds[0]:.6f}",
                "pred_max": f"{preds[-1]:.6f}",
            }
        )
    return rows


def _write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _volume_normalized_lift(made: float, exp_makes: float, shots: int, prior_shots: float) -> float:
    """
    Empirical-Bayes style shrinkage toward 0-lift.
    Equivalent to shrinking observed lift by n / (n + prior_shots).
    """
    return (made - exp_makes) / (shots + prior_shots)


def _load_shrink_shooter_3pt_priors(
    path: str, *, pseudo_attempts: float = 50.0
) -> Tuple[Dict[int, float], float]:
    """
    2025-26 regular season (from Oct 2025), Beta-style shrink toward league 3P%.
    Returns (person_id -> shrunk pct, league_pct).
    """
    tpm: Dict[int, float] = {}
    tpa: Dict[int, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("gameType") or "").strip() != "Regular Season":
                continue
            gd = (row.get("gameDate") or "").strip()
            if len(gd) >= 10 and gd[:10] < "2025-10-01":
                continue
            pid_raw = (row.get("personId") or "").strip()
            if not pid_raw:
                continue
            pid = int(pid_raw)
            ta = _to_float(row.get("threePointersAttempted", ""))
            tm = _to_float(row.get("threePointersMade", ""))
            if not math.isfinite(ta) or not math.isfinite(tm):
                continue
            tpa[pid] = tpa.get(pid, 0.0) + ta
            tpm[pid] = tpm.get(pid, 0.0) + tm

    league_tpa = sum(tpa.values())
    league_tpm = sum(tpm.values())
    league_pct = league_tpm / league_tpa if league_tpa > 1e-9 else 0.36

    priors: Dict[int, float] = {}
    k = pseudo_attempts
    for pid, att in tpa.items():
        made = tpm.get(pid, 0.0)
        priors[pid] = (made + k * league_pct) / (att + k)
    return priors, league_pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Model defender effectiveness as lift vs baseline.")
    parser.add_argument(
        "--csv",
        default="data/outputs/shot_contest_dataset.csv",
        help="Input shot dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/outputs",
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--min-defender-shots",
        type=int,
        default=5,
        help="Minimum shots to report a defender-level lift row.",
    )
    parser.add_argument(
        "--shrinkage-prior-shots",
        type=float,
        default=25.0,
        help="Pseudo-shot prior for volume-normalized defender lift.",
    )
    parser.add_argument(
        "--analysis-eligible-only",
        dest="analysis_eligible_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rows where analysis_eligible=yes (default: on; same pool as SCQ when it uses this filter). "
        "Pass --no-analysis-eligible-only to include all CSV rows that pass model completeness checks.",
    )
    parser.add_argument(
        "--defender-model-eligible-only",
        action="store_true",
        help="Keep rows where defender_model_eligible=yes (from build_unified_shot_dataset; "
        "aligns lift with SCQ driver tables). Overrides --analysis-eligible-only when both are set.",
    )
    parser.add_argument(
        "--player-statistics-csv",
        default="data/PlayerStatistics.csv",
        help="Used to fill shooter_2025_26_regular_3pt_pct when missing from shot CSV.",
    )
    parser.add_argument(
        "--heat-defender-lift-final-csv",
        default=None,
        help="Optional slim CSV: defender lift vs baseline (raw + volume-normalized).",
    )
    args = parser.parse_args()

    baseline_numeric_cols = [
        "shooter_2025_26_regular_3pt_pct",
        "shooter_dist_to_rim_in",
        "release_ball_z",
        "apex_ball_z",
        "release_shot_clock",
    ]
    # closeout_delta_ft_500ms excluded: it equals closeout_speed_ft_s * 0.5 (perfect collinearity).
    # shot_contest_quality excluded: it is a weighted sum of the four component features below,
    # creating near-perfect collinearity that prevents the model from separately identifying
    # component contributions.
    contest_numeric_cols = [
        "contest_distance_ft",
        "closeout_speed_ft_s",
        "contest_angle_deg",
        "hand_up_in",
    ]

    shots: List[Dict[str, str]] = []
    x_base_raw: List[List[float]] = []
    x_full_num_raw: List[List[float]] = []
    y: List[float] = []
    defender_ids: List[str] = []

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = (
            baseline_numeric_cols
            + contest_numeric_cols
            + ["pbp_shot_result", "nearest_defender_id", "nearest_defender_name", "shooter_id", "shooter_name"]
        )
        missing = [c for c in required if reader.fieldnames is None or c not in reader.fieldnames]
        merge_shooter = "shooter_2025_26_regular_3pt_pct" in missing
        if merge_shooter:
            missing = [c for c in missing if c != "shooter_2025_26_regular_3pt_pct"]
        if missing:
            raise ValueError(f"Missing required columns in input CSV: {missing}")

        if args.defender_model_eligible_only:
            if reader.fieldnames is None or "defender_model_eligible" not in reader.fieldnames:
                raise ValueError(
                    "--defender-model-eligible-only requires column defender_model_eligible; "
                    "re-run scripts/pipeline/build_unified_shot_dataset.py to refresh the shot CSV."
                )
        elif args.analysis_eligible_only:
            if reader.fieldnames is None or "analysis_eligible" not in reader.fieldnames:
                raise ValueError(
                    "analysis_eligible column is missing from the shot CSV. "
                    "Re-run scripts/pipeline/build_unified_shot_dataset.py, or pass --no-analysis-eligible-only."
                )

        priors: Optional[Dict[int, float]] = None
        league_pct = 0.36
        if merge_shooter:
            priors, league_pct = _load_shrink_shooter_3pt_priors(args.player_statistics_csv)

        for row in reader:
            if args.defender_model_eligible_only:
                if (row.get("defender_model_eligible") or "").strip().lower() != "yes":
                    continue
            elif args.analysis_eligible_only:
                if (row.get("analysis_eligible") or "").strip().lower() != "yes":
                    continue
            if merge_shooter and priors is not None:
                sid = (row.get("shooter_id") or "").strip()
                if not sid:
                    continue
                pct = priors.get(int(sid), league_pct)
                row["shooter_2025_26_regular_3pt_pct"] = f"{pct:.8f}"

            yi = _made_binary(row.get("pbp_shot_result", ""))
            if not math.isfinite(yi):
                continue
            base_vals = [_to_float(row.get(c, "")) for c in baseline_numeric_cols]
            contest_vals = [_to_float(row.get(c, "")) for c in contest_numeric_cols]
            if not all(math.isfinite(v) for v in base_vals + contest_vals):
                continue
            shots.append(row)
            x_base_raw.append(base_vals)
            x_full_num_raw.append(base_vals + contest_vals)
            y.append(yi)
            defender_ids.append((row.get("nearest_defender_id") or "").strip())

    if len(y) < 50:
        raise ValueError(f"Too few complete rows for stable model fit: {len(y)}")

    std_base = _fit_standardizer(x_base_raw)
    x_base = std_base.transform(x_base_raw)

    std_full_num = _fit_standardizer(x_full_num_raw)
    x_full_num = std_full_num.transform(x_full_num_raw)

    defender_map, def_ohe = _build_defender_ohe(defender_ids)
    x_full = [x_full_num[i] + def_ohe[i] for i in range(len(y))]

    base_model = RidgeLogit(lr=0.05, max_iter=7000, l2=2.0, seed=42)
    base_model.fit(x_base, y)
    p_base = base_model.predict_proba(x_base)

    full_model = RidgeLogit(lr=0.05, max_iter=7000, l2=6.0, seed=42)
    full_model.fit(x_full, y)
    p_full = full_model.predict_proba(x_full)

    print("Rows used:", len(y))
    print("Baseline logloss:", f"{_logloss(y, p_base):.4f}", "Brier:", f"{_brier(y, p_base):.4f}")
    print("Full logloss:", f"{_logloss(y, p_full):.4f}", "Brier:", f"{_brier(y, p_full):.4f}")
    print("Defender shrinkage terms:", len(defender_map), "(one defender dropped as reference)")

    # Marginal effects proxy: odds ratio for +1 SD in each standardized contest variable.
    start_contest = len(baseline_numeric_cols)
    print("\nContest feature effects (full model; odds ratio for +1 SD):")
    for j, col in enumerate(contest_numeric_cols):
        idx = start_contest + j
        beta = full_model.w[idx]
        odds_ratio = math.exp(beta)
        print(f"  {col}: beta={beta:+.4f}, OR(+1SD)={odds_ratio:.3f}")

    shot_rows: List[Dict[str, str]] = []
    for i, row in enumerate(shots):
        shot_rows.append(
            {
                "game_id": (row.get("game_id") or "").strip(),
                "shooter_id": (row.get("shooter_id") or "").strip(),
                "shooter_name": (row.get("shooter_name") or "").strip(),
                "nearest_defender_id": (row.get("nearest_defender_id") or "").strip(),
                "nearest_defender_name": (row.get("nearest_defender_name") or "").strip(),
                "made": str(int(y[i])),
                "p_base": f"{p_base[i]:.6f}",
                "p_full": f"{p_full[i]:.6f}",
                "lift_vs_baseline": f"{(y[i] - p_base[i]):.6f}",
                "lift_vs_full": f"{(y[i] - p_full[i]):.6f}",
            }
        )

    defender_aggs: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "made": 0.0, "exp_base": 0.0, "exp_full": 0.0, "var_base": 0.0}
    )
    for i, row in enumerate(shots):
        did = (row.get("nearest_defender_id") or "").strip()
        dnm = (row.get("nearest_defender_name") or "").strip()
        key = (did, dnm)
        defender_aggs[key]["n"] += 1.0
        defender_aggs[key]["made"] += y[i]
        defender_aggs[key]["exp_base"] += p_base[i]
        defender_aggs[key]["exp_full"] += p_full[i]
        defender_aggs[key]["var_base"] += p_base[i] * (1.0 - p_base[i])

    defender_rows: List[Dict[str, str]] = []
    for (did, dnm), s in sorted(defender_aggs.items(), key=lambda kv: kv[1]["n"], reverse=True):
        n = int(s["n"])
        if n < args.min_defender_shots:
            continue
        made = s["made"]
        exp_base = s["exp_base"]
        exp_full = s["exp_full"]
        var_base = max(s["var_base"], 1e-9)
        raw_lift_rate = (made - exp_base) / n
        raw_lift_makes = made - exp_base
        shrink_weight = n / (n + args.shrinkage_prior_shots)
        volume_norm_lift_rate = _volume_normalized_lift(
            made, exp_base, n, args.shrinkage_prior_shots
        )
        z_score = raw_lift_makes / math.sqrt(var_base)
        defender_rows.append(
            {
                "nearest_defender_id": did,
                "nearest_defender_name": dnm,
                "shots": str(n),
                "actual_make_rate": f"{made / n:.6f}",
                "baseline_expected_make_rate": f"{exp_base / n:.6f}",
                "full_expected_make_rate": f"{exp_full / n:.6f}",
                "lift_vs_baseline_rate": f"{raw_lift_rate:.6f}",
                "lift_vs_full_rate": f"{(made - exp_full) / n:.6f}",
                "volume_normalized_lift_rate": f"{volume_norm_lift_rate:.6f}",
                "shrinkage_weight": f"{shrink_weight:.6f}",
                "lift_vs_baseline_z": f"{z_score:.6f}",
                "actual_minus_expected_makes_baseline": f"{made - exp_base:.6f}",
                "actual_minus_expected_makes_full": f"{made - exp_full:.6f}",
            }
        )

    cal_base = _calibration_bins(y, p_base, n_bins=10)
    cal_full = _calibration_bins(y, p_full, n_bins=10)

    out_shots = f"{args.out_dir}/defensive_effectiveness_shot_level.csv"
    out_defs = f"{args.out_dir}/defender_effectiveness_lift.csv"
    out_cal_base = f"{args.out_dir}/calibration_baseline.csv"
    out_cal_full = f"{args.out_dir}/calibration_full.csv"

    _write_csv(
        out_shots,
        [
            "game_id",
            "shooter_id",
            "shooter_name",
            "nearest_defender_id",
            "nearest_defender_name",
            "made",
            "p_base",
            "p_full",
            "lift_vs_baseline",
            "lift_vs_full",
        ],
        shot_rows,
    )
    _write_csv(
        out_defs,
        [
            "nearest_defender_id",
            "nearest_defender_name",
            "shots",
            "actual_make_rate",
            "baseline_expected_make_rate",
            "full_expected_make_rate",
            "lift_vs_baseline_rate",
            "lift_vs_full_rate",
            "volume_normalized_lift_rate",
            "shrinkage_weight",
            "lift_vs_baseline_z",
            "actual_minus_expected_makes_baseline",
            "actual_minus_expected_makes_full",
        ],
        defender_rows,
    )
    _write_csv(out_cal_base, ["bin", "n", "pred_mean", "obs_rate", "pred_min", "pred_max"], cal_base)
    _write_csv(out_cal_full, ["bin", "n", "pred_mean", "obs_rate", "pred_min", "pred_max"], cal_full)

    if args.heat_defender_lift_final_csv:
        slim: List[Dict[str, str]] = []
        for r in defender_rows:
            slim.append(
                {
                    "nearest_defender_id": r["nearest_defender_id"],
                    "nearest_defender_name": r["nearest_defender_name"],
                    "shots": r["shots"],
                    "actual_make_rate": r["actual_make_rate"],
                    "baseline_expected_make_rate": r["baseline_expected_make_rate"],
                    "raw_lift_vs_baseline": r["lift_vs_baseline_rate"],
                    "volume_normalized_lift": r["volume_normalized_lift_rate"],
                    "shrinkage_prior_shots": f"{args.shrinkage_prior_shots:.1f}",
                }
            )
        _write_csv(
            args.heat_defender_lift_final_csv,
            [
                "nearest_defender_id",
                "nearest_defender_name",
                "shots",
                "actual_make_rate",
                "baseline_expected_make_rate",
                "raw_lift_vs_baseline",
                "volume_normalized_lift",
                "shrinkage_prior_shots",
            ],
            slim,
        )

    print("\nWrote:")
    print(" ", out_shots)
    print(" ", out_defs)
    print(" ", out_cal_base)
    print(" ", out_cal_full)
    if args.heat_defender_lift_final_csv:
        print(" ", args.heat_defender_lift_final_csv)


if __name__ == "__main__":
    main()
