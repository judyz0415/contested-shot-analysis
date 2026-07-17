#!/usr/bin/env python3
"""
calibrate_action_model.py

Multi-class calibration diagnostics for the Action-3D snapshot model.

Steps:
  1. Re-run 5-fold GroupKFold CV to get OOF predicted probabilities.
  2. Save OOF probs + true labels to data/outputs/action_model_oof_predictions.csv
  3. For each of the 5 action classes, bin P(class) into 10 equal-frequency deciles.
  4. Compute ECE_k = (1/N) * sum_bins[ n_bin * |mean_pred - obs_freq| ]
  5. Save per-bin calibration table to data/outputs/action_model_calibration.csv
  6. Print per-class ECE and overall weighted ECE.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

TOUCH_CSV = "data/intermediate/touch_events_combined.csv"
OOF_CSV   = "data/outputs/action_model_oof_predictions.csv"
CAL_CSV   = "data/outputs/action_model_calibration.csv"

F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
            "release_shot_clock", "game_clock_seconds"]
POSE    = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
           "lateral_velocity_ft_s", "min_def_joint_dist_ft"]
F3D     = F2D_CTX + POSE
REF     = "pass_kickout"
N_BINS  = 10


def load(path):
    df = pd.read_csv(path)
    for c in F3D:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    for c in F3D:
        df[c] = df[c].fillna(df[c].mean())
    return df


def run_oof_cv(df, classes):
    """5-fold GroupKFold CV; returns OOF probability matrix (n, K)."""
    X      = df[F3D].values
    y      = df["action"].values
    groups = df["game_file"].values
    oof    = np.zeros((len(df), len(classes)))

    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
        sc  = StandardScaler().fit(X[tr])
        m   = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800)
        m.fit(sc.transform(X[tr]), y[tr])
        # align columns to fixed class order
        col_idx = [list(m.classes_).index(c) for c in classes]
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, col_idx]
        print(f"  fold {fold+1}/5  ({len(te)} test obs, {len(tr)} train obs)")

    return oof


def calibration_per_class(y_true, oof, classes, n_bins=10):
    """
    For each class, bin P(class) into n_bins equal-frequency bins.
    Returns a long-form DataFrame with columns:
      class_label, bin, n, mean_pred_prob, obs_freq, abs_error
    """
    N = len(y_true)
    rows = []
    ece_per_class = {}

    for ki, cls in enumerate(classes):
        p_pred = oof[:, ki]
        y_obs  = (y_true == cls).astype(float)

        # equal-frequency bin edges (quantile-based)
        edges = np.percentile(p_pred, np.linspace(0, 100, n_bins + 1))
        edges[0]  -= 1e-9   # include minimum
        edges[-1] += 1e-9   # include maximum
        bin_ids = np.digitize(p_pred, edges[1:-1])   # 0 … n_bins-1

        ece_accum = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            n_b  = mask.sum()
            if n_b == 0:
                continue
            mp  = float(p_pred[mask].mean())
            of  = float(y_obs[mask].mean())
            err = abs(mp - of)
            ece_accum += n_b * err
            rows.append({
                "class_label":    cls,
                "bin":            b + 1,
                "n":              int(n_b),
                "mean_pred_prob": round(mp, 6),
                "obs_freq":       round(of, 6),
                "abs_error":      round(err, 6),
            })

        ece_per_class[cls] = ece_accum / N

    return pd.DataFrame(rows), ece_per_class


def main():
    df      = load(TOUCH_CSV)
    classes = sorted(df["action"].unique())
    print(f"n = {len(df)}  |  {df.game_file.nunique()} games  |  classes: {classes}\n")

    # ── 1. OOF CV ────────────────────────────────────────────────────────────
    print("Running 5-fold GroupKFold CV …")
    oof = run_oof_cv(df, classes)

    # ── 2. Save OOF predictions ───────────────────────────────────────────────
    oof_df = pd.DataFrame(oof, columns=[f"p_{c}" for c in classes])
    oof_df.insert(0, "true_label", df["action"].values)
    oof_df.insert(1, "game_file",  df["game_file"].values)
    oof_df.to_csv(OOF_CSV, index=False)
    print(f"\nSaved OOF predictions → {OOF_CSV}  ({len(oof_df)} rows)\n")

    # ── 3 & 4. Calibration per class ─────────────────────────────────────────
    cal_df, ece_per_class = calibration_per_class(
        df["action"].values, oof, classes, n_bins=N_BINS
    )

    # ── 5. Save calibration table ─────────────────────────────────────────────
    cal_df.to_csv(CAL_CSV, index=False)
    print(f"Saved calibration table → {CAL_CSV}\n")

    # ── 6. Print per-bin detail ───────────────────────────────────────────────
    print("── Per-class calibration (predicted decile vs. observed rate) ───────")
    for cls in classes:
        sub = cal_df[cal_df.class_label == cls]
        base = (df["action"] == cls).mean()
        print(f"\n  {cls}  (base rate {base:.3f}  |  ECE = {ece_per_class[cls]:.5f})")
        print(f"  {'Bin':>4}  {'N':>6}  {'MeanPred':>10}  {'ObsFreq':>9}  {'|err|':>7}")
        for _, r in sub.iterrows():
            flag = "  !!!" if r.abs_error > 0.05 else ""
            print(f"  {int(r.bin):>4}  {int(r.n):>6}  {r.mean_pred_prob:>10.4f}  "
                  f"{r.obs_freq:>9.4f}  {r.abs_error:>7.4f}{flag}")

    # ── 7. ECE summary table ──────────────────────────────────────────────────
    N     = len(df)
    total = sum(df["action"].value_counts()[c] * ece_per_class[c] for c in classes) / N
    print("\n── ECE Summary ─────────────────────────────────────────────────────")
    print(f"  {'Class':<22}  {'Base rate':>10}  {'ECE':>8}")
    print(f"  {'─'*44}")
    for cls in classes:
        base = (df["action"] == cls).mean()
        print(f"  {cls:<22}  {base:>10.4f}  {ece_per_class[cls]:>8.5f}")
    print(f"  {'─'*44}")
    print(f"  {'Weighted overall ECE':<22}  {'':>10}  {total:>8.5f}")


if __name__ == "__main__":
    main()
