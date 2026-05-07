#!/usr/bin/env python3
"""
Evaluate whether an alteredness metric is redundant with existing SCQ features
and whether it behaves like a richer contestedness signal than make/miss.

Pure stdlib (no NumPy): works with the system `python3` on macOS.

Primary questions answered:
1) Is alteredness mostly explained by existing contest features?
2) Does alteredness add signal beyond existing features for shot outcome?
3) Is alteredness less noisy / more continuous than make-miss as a pressure proxy?

Usage:
    python scripts/analyze_alteredness_metric.py \\
      --csv data/outputs/shot_contest_dataset.csv \\
      --alteredness-col alteredness
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Sequence, Tuple

DEFAULT_FEATURES = [
    "contest_distance_ft",
    "closeout_speed_ft_s",
    "contest_angle_deg",
    "hand_up_in",
]


def _to_float(v: str) -> float:
    s = (v or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _shot_result_to_binary(v: str) -> float:
    s = (v or "").strip().lower()
    if not s or s in {"na", "nan", "none"}:
        return float("nan")
    if "made" in s:
        return 1.0
    if "miss" in s:
        return 0.0
    return float("nan")


def _solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivot (a: n x n, b: length n)."""
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Singular or near-singular normal equations")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        piv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= piv
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= f * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _ols_fit(x: Sequence[Sequence[float]], y: Sequence[float]) -> Tuple[List[float], List[float], float]:
    """Ordinary least squares with intercept: y ~ [1, x]. Returns beta, yhat, r2."""
    n = len(y)
    if n == 0:
        raise ValueError("Empty regression")
    p = len(x[0])
    p1 = p + 1
    xt_x = [[0.0 for _ in range(p1)] for _ in range(p1)]
    xty = [0.0 for _ in range(p1)]
    x1_rows: List[List[float]] = []
    for i in range(n):
        row = [1.0] + list(x[i])
        x1_rows.append(row)

    for i in range(n):
        xi = x1_rows[i]
        yi = float(y[i])
        for a in range(p1):
            for b in range(p1):
                xt_x[a][b] += xi[a] * xi[b]
            xty[a] += xi[a] * yi

    beta = _solve_linear(xt_x, xty)

    mean_y = sum(y) / n
    sse = 0.0
    sst = 0.0
    yhat: List[float] = []
    for i in range(n):
        yi = float(y[i])
        pred = sum(beta[j] * x1_rows[i][j] for j in range(p1))
        yhat.append(pred)
        sse += (yi - pred) ** 2
        sst += (yi - mean_y) ** 2
    r2 = 0.0 if sst <= 0 else max(0.0, 1.0 - sse / sst)
    return beta, yhat, r2


def _kfold_r2(x: List[List[float]], y: List[float], k: int = 5, seed: int = 42) -> float:
    n = len(y)
    if n < max(k, 10):
        return float("nan")
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    folds: List[List[int]] = []
    start = 0
    for fs in fold_sizes:
        folds.append(idx[start : start + fs])
        start += fs

    r2s: List[float] = []
    for ti in range(k):
        test_idx = folds[ti]
        train_idx = [idx for j, fd in enumerate(folds) if j != ti for idx in fd]
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        x_tr = [x[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        beta, _, _ = _ols_fit(x_tr, y_tr)
        p = len(x[0])
        p1 = p + 1
        mean_te = sum(y[i] for i in test_idx) / len(test_idx)
        sse = sst = 0.0
        for i in test_idx:
            xi = [1.0] + list(x[i])
            pred = sum(beta[j] * xi[j] for j in range(p1))
            yi = y[i]
            sse += (yi - pred) ** 2
            sst += (yi - mean_te) ** 2
        if sst <= 0:
            continue
        r2s.append(1.0 - sse / sst)
    return sum(r2s) / len(r2s) if r2s else float("nan")


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = denx = deny = 0.0
    for i in range(len(xs)):
        dx = xs[i] - mx
        dy = ys[i] - my
        num += dx * dy
        denx += dx * dx
        deny += dy * dy
    if denx <= 0.0 or deny <= 0.0:
        return float("nan")
    return num / math.sqrt(denx * deny)


def _drop_col(x: List[List[float]], col: int) -> List[List[float]]:
    return [row[:col] + row[col + 1 :] for row in x]


def _load_rows(
    csv_path: str,
    feature_cols: Sequence[str],
    alteredness_col: str,
    outcome_col: str,
) -> Dict[str, object]:
    x_alt: List[List[float]] = []
    y_alt: List[float] = []

    x_out: List[List[float]] = []
    y_out: List[float] = []
    a_out: List[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = list(feature_cols) + [alteredness_col, outcome_col]
        missing = [c for c in required if reader.fieldnames is None or c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        assert reader.fieldnames is not None
        for row in reader:
            feats = [_to_float(row[c]) for c in feature_cols]
            alt = _to_float(row[alteredness_col])
            out = _shot_result_to_binary(row[outcome_col])

            if all(math.isfinite(v) for v in feats) and math.isfinite(alt):
                x_alt.append(feats)
                y_alt.append(alt)

            if all(math.isfinite(v) for v in feats) and math.isfinite(out):
                x_out.append(feats)
                y_out.append(out)
                a_out.append(alt if math.isfinite(alt) else float("nan"))

    return {
        "x_alt": x_alt,
        "y_alt": y_alt,
        "x_out": x_out,
        "y_out": y_out,
        "a_out": a_out,
    }


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _interpret_redundancy(r2: float) -> str:
    if not math.isfinite(r2):
        return "insufficient data"
    if r2 >= 0.70:
        return "mostly explained by existing features (high redundancy)"
    if r2 >= 0.40:
        return "partly explained (moderate redundancy)"
    return "largely independent from current features (low redundancy)"


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose alteredness vs existing SCQ features and shot outcome."
    )
    parser.add_argument(
        "--csv",
        default="data/outputs/shot_contest_dataset.csv",
        help="Input dataset CSV (default: data/outputs/shot_contest_dataset.csv)",
    )
    parser.add_argument(
        "--alteredness-col",
        required=True,
        help="Column name for alteredness metric (numeric).",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Existing contest feature columns to test against.",
    )
    parser.add_argument(
        "--outcome-col",
        default="pbp_shot_result",
        help="Shot outcome column (default: pbp_shot_result; expects made/miss text).",
    )
    args = parser.parse_args()

    data = _load_rows(
        csv_path=args.csv,
        feature_cols=args.feature_cols,
        alteredness_col=args.alteredness_col,
        outcome_col=args.outcome_col,
    )

    x_alt = data["x_alt"]  # type: ignore[assignment]
    y_alt = data["y_alt"]  # type: ignore[assignment]
    x_out = data["x_out"]  # type: ignore[assignment]
    y_out = data["y_out"]  # type: ignore[assignment]
    a_out = data["a_out"]  # type: ignore[assignment]

    if len(y_alt) < 20:
        raise ValueError("Too few usable rows for alteredness analysis (<20).")
    if len(y_out) < 20:
        raise ValueError("Too few made/miss rows for outcome analysis (<20).")

    _, alt_yhat, alt_r2 = _ols_fit(x_alt, y_alt)
    alt_cv_r2 = _kfold_r2(x_alt, y_alt, k=5, seed=42)

    _print_header("1) Alteredness explained by existing SCQ features")
    print(f"rows_used: {len(y_alt)}")
    print(f"in_sample_r2: {alt_r2:.3f}")
    print(f"5fold_cv_r2: {alt_cv_r2:.3f}" if math.isfinite(alt_cv_r2) else "5fold_cv_r2: nan")
    print(f"interpretation: {_interpret_redundancy(alt_r2)}")
    _print_header("Feature-specific contribution to alteredness (drop in R2)")
    for j, feat in enumerate(args.feature_cols):
        x_loo = _drop_col(x_alt, j)
        _, _, loo_r2 = _ols_fit(x_loo, y_alt)
        print(f"{feat}: r2_drop={alt_r2 - loo_r2:.3f}")

    _, _, mm_r2 = _ols_fit(x_out, y_out)
    mm_cv_r2 = _kfold_r2(x_out, y_out, k=5, seed=42)

    _print_header("2) Baseline explainability by defense context")
    print(f"made_miss_rows: {len(y_out)}")
    print(f"R2(alteredness ~ features): {alt_r2:.3f}")
    print(f"R2(make_miss ~ features):  {mm_r2:.3f}")
    print(
        "If alteredness R2 is materially higher than make/miss R2, "
        "alteredness is likely a less noisy and more continuous pressure target."
    )
    print(
        f"5fold_cv_r2(make_miss ~ features): {mm_cv_r2:.3f}"
        if math.isfinite(mm_cv_r2)
        else "5fold_cv_r2(make_miss ~ features): nan"
    )

    pairs = [
        (
            list(x_out[i])
            + [a_out[i]],
            y_out[i],
        )
        for i in range(len(y_out))
        if math.isfinite(a_out[i])
    ]

    _print_header("3) Incremental value for make/miss")
    if len(pairs) >= 20:
        x_joint = [p[0][:-1] for p in pairs]
        alt_col = [p[0][-1] for p in pairs]
        y_joint = [p[1] for p in pairs]
        _, _, base_r2 = _ols_fit(x_joint, y_joint)

        xp1 = [[*row, alt_col[m]] for m, row in enumerate(x_joint)]
        _, _, plus_r2 = _ols_fit(xp1, y_joint)
        print(f"rows_with_alteredness_and_outcome: {len(pairs)}")
        print(f"R2(make_miss ~ features):               {base_r2:.3f}")
        print(f"R2(make_miss ~ features + alteredness): {plus_r2:.3f}")
        print(f"delta_r2_from_alteredness:              {plus_r2 - base_r2:.3f}")
        c = _corr(alt_col, y_joint)
        print(f"corr(alteredness, make):                {c:.3f}" if math.isfinite(c) else "corr(alteredness, make): nan")
        print(
            "Small positive delta_r2 means alteredness has non-redundant signal "
            "linked to outcome, beyond existing contest geometry."
        )
    else:
        print("Not enough rows with both alteredness and made/miss labels.")

    resid_alt = [y_alt[i] - alt_yhat[i] for i in range(len(y_alt))]
    mr = sum(resid_alt) / len(resid_alt)
    vr = math.sqrt(sum((r - mr) ** 2 for r in resid_alt) / max(1, (len(resid_alt) - 1)))

    _print_header("4) Recommended residual metric")
    print("Create alteredness_resid = alteredness - E[alteredness | existing_features].")
    print("This strips off the part already implied by current SCQ inputs.")
    print(f"residual_std: {vr:.3f}")
    print(f"residual_mean: {mr:.3f}")

    _print_header("Decision guide")
    print("- High R2 + tiny delta_r2: alteredness is mostly re-packaging existing features.")
    print("- Moderate/low R2 + positive delta_r2: alteredness likely adds meaningful information.")
    print("- Prefer residualized alteredness in SCQ to avoid double counting pressure cues.")


if __name__ == "__main__":
    _main()
