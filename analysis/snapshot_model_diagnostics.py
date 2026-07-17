#!/usr/bin/env python3
"""
snapshot_model_diagnostics.py

Three diagnostic tests on the Action-3D snapshot model:

  1. Calibration  — per-class predicted-probability deciles vs. observed rates
  2. Hausman-McFadden IIA — drop one alternative at a time, test coefficient shift
  3. VIF for the pose feature block (check lateral_velocity / min_def_joint_dist_ft collinearity)
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

TOUCH_CSV = "data/intermediate/touch_events_combined.csv"
F2D  = ["def_dist_at_catch_ft", "n_open_3pt_threats",
        "min_kickout_defender_gap_ft", "release_shot_clock", "game_clock_seconds"]
F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
            "release_shot_clock", "game_clock_seconds"]
POSE = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
        "lateral_velocity_ft_s", "min_def_joint_dist_ft"]
F3D  = F2D_CTX + POSE
REF  = "pass_kickout"


def load(path):
    df = pd.read_csv(path)
    for c in F3D:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    for c in F3D:
        df[c] = df[c].fillna(df[c].mean())
    return df


def fit_mnlogit(X_z, yc, maxiter=400):
    return sm.MNLogit(yc, sm.add_constant(X_z)).fit(disp=0, maxiter=maxiter)


# ── 1. Calibration ───────────────────────────────────────────────────────────
def calibration(df, classes, sc, m):
    """Bin predicted P(class) into deciles; compare to observed rates."""
    X_c = sm.add_constant(sc.transform(df[F3D]))
    probs = m.predict(X_c)           # statsmodels returns (n, K) including ref
    # column order = categorical codes: ref=0, then non-ref in definition order
    non_ref   = [c for c in classes if c != REF]
    col_order = [REF] + non_ref      # matches codes 0..K-1
    prob_df   = pd.DataFrame(probs, columns=col_order)

    print("\n── 1. Calibration: predicted decile vs. observed rate ───────────────")
    rows = []
    for cls in classes:
        p_pred = prob_df[cls].values
        y_obs  = (df.action == cls).astype(float).values
        edges  = np.percentile(p_pred, np.linspace(0, 100, 11))
        edges[0] -= 1e-9             # include lowest value
        bin_ids = np.digitize(p_pred, edges[1:])   # 0..9

        print(f"\n  {cls}  (base rate {y_obs.mean():.3f})")
        print(f"  {'Bin':>4}  {'Pred':>8}  {'Obs':>8}  {'N':>6}  {'|err|':>7}")
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            pm  = p_pred[mask].mean()
            om  = y_obs[mask].mean()
            n   = mask.sum()
            err = abs(pm - om)
            flag = "  !!!" if err > 0.05 else ""
            print(f"  {b+1:>4}  {pm:>8.4f}  {om:>8.4f}  {n:>6}  {err:>7.4f}{flag}")
            rows.append({"class": cls, "bin": b + 1,
                         "mean_pred": round(pm, 5), "obs_rate": round(om, 5),
                         "n": int(n), "abs_err": round(err, 5)})
    return pd.DataFrame(rows)


# ── 2. Hausman-McFadden IIA ──────────────────────────────────────────────────
def hausman_mcfadden(df, classes, sc):
    """
    For each non-reference alternative, drop it + its observations, refit the
    restricted model, and compute the Hausman-McFadden chi-squared statistic.

    Under IIA, dropping an irrelevant alternative should not shift coefficients
    for the remaining classes. A significant H suggests IIA violation.
    """
    non_ref_full = [c for c in classes if c != REF]
    X_full       = sc.transform(df[F3D])
    ycat_f       = pd.Categorical(df.action, categories=[REF] + non_ref_full)
    m_f          = fit_mnlogit(X_full, ycat_f.codes)
    params_f     = np.array(m_f.params)      # (n_params, K-1)
    cov_f        = np.array(m_f.cov_params()) # (n_params*(K-1), n_params*(K-1))
    n_params     = X_full.shape[1] + 1       # features + constant

    print("\n── 2. Hausman-McFadden IIA test ─────────────────────────────────────")
    print(f"  {'Dropped alt':<22}  {'H stat':>9}  {'df':>5}  {'p':>10}  sig  note")
    rows = []
    for dropped in non_ref_full:
        non_ref_r = [c for c in non_ref_full if c != dropped]
        df_r      = df[df.action != dropped].copy()
        ycat_r    = pd.Categorical(df_r.action, categories=[REF] + non_ref_r)
        X_r       = sc.transform(df_r[F3D])
        m_r       = fit_mnlogit(X_r, ycat_r.codes)
        params_r  = np.array(m_r.params)
        cov_r     = np.array(m_r.cov_params())

        # Indices of common alternatives in the full model parameter matrix
        full_alt_idx = [non_ref_full.index(c) for c in non_ref_r]  # column indices

        # Extract coefficient vectors (vectorised column-major: params for alt0, then alt1, …)
        b_f = params_f[:, full_alt_idx].flatten(order='F')
        b_r = params_r.flatten(order='F')
        b_diff = b_r - b_f

        # Build subblock of full covariance matching the common alternatives
        row_idx = []
        for ai in full_alt_idx:
            row_idx.extend(range(ai * n_params, (ai + 1) * n_params))
        cov_f_sub = cov_f[np.ix_(row_idx, row_idx)]

        V_diff = cov_r - cov_f_sub

        # Check PSD; use pseudoinverse with negative-eigenvalue warning
        eigvals = np.linalg.eigvalsh(V_diff)
        n_neg   = (eigvals < -1e-8).sum()
        note    = f"{n_neg} neg eigvals → pinv" if n_neg > 0 else "PSD ok"
        V_inv   = np.linalg.pinv(V_diff)
        H       = float(b_diff @ V_inv @ b_diff)
        df_stat = len(b_diff)
        p       = stats.chi2.sf(H, df_stat) if H > 0 else 1.0
        sig     = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {dropped:<22}  {H:>9.2f}  {df_stat:>5}  {p:>10.4f}  {sig:<3}  {note}")
        rows.append({"dropped": dropped, "H": round(H, 3), "df": df_stat,
                     "p": round(p, 6), "neg_eigvals": int(n_neg)})
    return pd.DataFrame(rows)


# ── 3. VIF ───────────────────────────────────────────────────────────────────
def compute_vif(df, feats, label):
    sc  = StandardScaler().fit(df[feats])
    X   = sc.transform(df[feats])
    vif = [variance_inflation_factor(X, i) for i in range(len(feats))]
    print(f"\n  {label}:")
    print(f"  {'Feature':<32}  {'VIF':>6}")
    rows = []
    for f, v in zip(feats, vif):
        flag = "  <<< HIGH (>5)" if v > 5 else ("  < note (>2.5)" if v > 2.5 else "")
        print(f"  {f:<32}  {v:>6.2f}{flag}")
        rows.append({"feature": f, "VIF": round(v, 3)})
    return pd.DataFrame(rows)


def corr_matrix(df, feats, label):
    print(f"\n  Pearson correlation — {label}:")
    corr = df[feats].corr().round(3)
    print(corr.to_string())


def main():
    df      = load(TOUCH_CSV)
    classes = sorted(df.action.unique())
    non_ref = [c for c in classes if c != REF]
    sc3     = StandardScaler().fit(df[F3D])
    ycat    = pd.Categorical(df.action, categories=[REF] + non_ref)
    yc      = ycat.codes
    m3      = fit_mnlogit(sc3.transform(df[F3D]), yc)

    print(f"n = {len(df)}  |  {df.game_file.nunique()} games  |  {len(classes)} classes")

    # 1. Calibration
    cal_df = calibration(df, classes, sc3, m3)
    cal_df.to_csv("data/outputs/calibration_check.csv", index=False)
    print("\n  → wrote calibration_check.csv")

    # 2. Hausman-McFadden
    hm_df = hausman_mcfadden(df, classes, sc3)
    hm_df.to_csv("data/outputs/hausman_mcfadden.csv", index=False)
    print("  → wrote hausman_mcfadden.csv")

    # 3. VIF + correlation
    print("\n── 3. VIF and correlation — pose feature block ──────────────────────")
    vif_pose = compute_vif(df, POSE, "Pose features")
    vif_full = compute_vif(df, F3D, "Full F3D set")
    corr_matrix(df, POSE, "pose features")
    vif_pose.to_csv("data/outputs/vif_pose.csv", index=False)
    vif_full.to_csv("data/outputs/vif_f3d.csv", index=False)
    print("\n  → wrote vif_pose.csv, vif_f3d.csv")


if __name__ == "__main__":
    main()
