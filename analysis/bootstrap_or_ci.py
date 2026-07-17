#!/usr/bin/env python3
"""
bootstrap_or_ci.py

Bootstrap 95% CIs on Action-3D odds ratios, resampling by GAME to respect
within-game clustering. B=500 iterations; uses sklearn LogisticRegression
(same spec as main model: lbfgs, C=1.0) for speed.

OR = exp(coef_k - coef_ref) per +1 SD, re-z-scored on each bootstrap sample.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TOUCH_CSV = "data/intermediate/touch_events_combined.csv"
OUT_CSV   = "data/outputs/action_model_bootstrap_or_ci.csv"

F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
            "release_shot_clock", "game_clock_seconds"]
POSE    = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
           "lateral_velocity_ft_s", "min_def_joint_dist_ft"]
F3D     = F2D_CTX + POSE
REF     = "pass_kickout"
B       = 500
RNG     = np.random.default_rng(42)


def load(path):
    df = pd.read_csv(path)
    for c in F3D:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    for c in F3D:
        df[c] = df[c].fillna(df[c].mean())
    return df


def fit_lr(X, y):
    return LogisticRegression(solver="lbfgs", C=1.0, max_iter=800).fit(X, y)


def extract_or(m, classes, feats):
    """OR = exp(beta_k - beta_ref) per feature × non-ref class."""
    cls_order = list(m.classes_)
    ref_i     = cls_order.index(REF)
    ors = {}
    for c in classes:
        if c == REF:
            continue
        ci = cls_order.index(c)
        for fi, f in enumerate(feats):
            beta        = m.coef_[ci, fi] - m.coef_[ref_i, fi]
            ors[(f, c)] = float(np.exp(beta))
    return ors


def main():
    df      = load(TOUCH_CSV)
    classes = sorted(df["action"].unique())
    games   = df["game_file"].unique()
    n_games = len(games)
    print(f"n = {len(df)}  |  {n_games} games  |  B = {B} bootstrap iterations\n")

    # ── Point estimate (full dataset) ────────────────────────────────────────
    sc_full   = StandardScaler().fit(df[F3D])
    X_full    = sc_full.transform(df[F3D])
    y_full    = df["action"].values
    m_full    = fit_lr(X_full, y_full)
    point_ors = extract_or(m_full, classes, F3D)

    # ── Bootstrap ────────────────────────────────────────────────────────────
    boot_store: dict[tuple, list] = {k: [] for k in point_ors}
    n_failed = 0

    print("Running bootstrap …")
    for b in range(B):
        boot_games = RNG.choice(games, size=n_games, replace=True)
        frames     = [df[df["game_file"] == g] for g in boot_games]
        boot_df    = pd.concat(frames, ignore_index=True)

        sc_b = StandardScaler().fit(boot_df[F3D])
        X_b  = sc_b.transform(boot_df[F3D])
        y_b  = boot_df["action"].values

        # skip if any class is missing (rare with 15 games but possible)
        if len(np.unique(y_b)) < len(classes):
            n_failed += 1
            continue
        try:
            m_b   = fit_lr(X_b, y_b)
            ors_b = extract_or(m_b, classes, F3D)
            for k, v in ors_b.items():
                boot_store[k].append(v)
        except Exception:
            n_failed += 1

        if (b + 1) % 100 == 0:
            print(f"  {b+1}/{B}  (failed: {n_failed})")

    print(f"\nDone. {B - n_failed}/{B} iterations succeeded.\n")

    # ── Compile results ───────────────────────────────────────────────────────
    rows = []
    for (feat, cls), vals in boot_store.items():
        arr = np.array(vals)
        rows.append({
            "feature":    feat,
            "class_label": cls,
            "or_point":   round(point_ors[(feat, cls)], 4),
            "ci_lower":   round(float(np.percentile(arr, 2.5)),  4),
            "ci_upper":   round(float(np.percentile(arr, 97.5)), 4),
            "n_boot":     len(arr),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}\n")

    # ── Print pose feature table ──────────────────────────────────────────────
    non_ref = [c for c in classes if c != REF]
    print("── Pose features: OR [95% bootstrap CI] vs pass/kickout ─────────────")
    header = f"  {'Feature':<32}" + "".join(f"  {c[:14]:>18}" for c in non_ref)
    print(header)
    print("  " + "─" * (32 + 20 * len(non_ref)))

    for feat in POSE:
        row_str = f"  {feat:<32}"
        for cls in non_ref:
            sub = result[(result.feature == feat) & (result.class_label == cls)]
            if sub.empty:
                row_str += f"  {'N/A':>18}"
                continue
            r = sub.iloc[0]
            sig = "*" if not (r.ci_lower <= 1.0 <= r.ci_upper) else " "
            cell = f"{r.or_point:.3f} [{r.ci_lower:.3f},{r.ci_upper:.3f}]{sig}"
            row_str += f"  {cell:>18}"
        print(row_str)

    print()
    print("  * = 95% CI excludes 1.0  (CI does not cross null)")

    # ── Also print F2D_CTX features for completeness ─────────────────────────
    print("\n── Context features: OR [95% bootstrap CI] vs pass/kickout ──────────")
    print(header)
    print("  " + "─" * (32 + 20 * len(non_ref)))
    for feat in F2D_CTX:
        row_str = f"  {feat:<32}"
        for cls in non_ref:
            sub = result[(result.feature == feat) & (result.class_label == cls)]
            if sub.empty:
                row_str += f"  {'N/A':>18}"
                continue
            r = sub.iloc[0]
            sig = "*" if not (r.ci_lower <= 1.0 <= r.ci_upper) else " "
            cell = f"{r.or_point:.3f} [{r.ci_lower:.3f},{r.ci_upper:.3f}]{sig}"
            row_str += f"  {cell:>18}"
        print(row_str)

    print()
    print("  * = 95% CI excludes 1.0")


if __name__ == "__main__":
    main()
