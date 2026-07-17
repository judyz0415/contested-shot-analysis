#!/usr/bin/env python3
"""
action_probability_model.py

P(action | closeout_state) on perimeter touches.

Tests whether 3D pose features improve action prediction beyond 2D floor
context (Action-2D vs Action-3D), using a multinomial logistic regression.

Default input: touch_events_combined.csv (Heat offense + Heat defense pooled,
~6,905 touches, treating each touch as an independent sample regardless of
which team is handling).

Outputs:
  data/outputs/action_probability_2d.csv          Action-2D coefficients
  data/outputs/action_probability_3d.csv          Action-3D coefficients
  data/outputs/action_probability_predictions.csv predicted distributions
  data/outputs/action_probability_per_feature_lrt.csv per-pose-feature LRTs
"""
from __future__ import annotations
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

F2D  = ["def_dist_at_catch_ft", "n_open_3pt_threats",
        "min_kickout_defender_gap_ft", "release_shot_clock", "game_clock_seconds"]
# F3D replaces 2D centroid distance with 3D joint distance; does not stack both
F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
            "release_shot_clock", "game_clock_seconds"]
POSE = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
        "lateral_velocity_ft_s", "min_def_joint_dist_ft"]
F3D  = F2D_CTX + POSE
REF   = "pass_kickout"
SHOTS = ("catch_and_shoot", "contested_shot")


def cv_logloss(df, feats, y, classes, groups):
    X = StandardScaler().fit_transform(df[feats])
    oof = np.zeros((len(df), len(classes)))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800).fit(X[tr], y[tr])
        idx = [list(m.classes_).index(c) for c in classes]
        oof[te] = m.predict_proba(X[te])[:, idx]
    return log_loss(y, oof, labels=classes)


def mnlogit(X, yc, maxiter=300):
    return sm.MNLogit(yc, sm.add_constant(X)).fit(disp=0, maxiter=maxiter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", default="data/intermediate/touch_events_combined.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.touches)
    for c in F3D:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    for c in F3D:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    n, ng = len(df), df.game_file.nunique()
    print(f"n = {n} touches, {ng} games")
    print("action counts:", df.action.value_counts().to_dict())

    y      = df.action.values
    groups = df.game_file.values
    classes = sorted(df.action.unique())
    K = len(classes)

    ycat = pd.Categorical(df.action, categories=[REF] + [c for c in classes if c != REF])
    yc   = ycat.codes
    sc2  = StandardScaler().fit(df[F2D])
    sc3  = StandardScaler().fit(df[F3D])
    X2z  = sc2.transform(df[F2D])
    X3z  = sc3.transform(df[F3D])

    # ── 1. OOF log-loss ────────────────────────────────────────────────────
    ll2  = cv_logloss(df, F2D, y, classes, groups)
    ll3  = cv_logloss(df, F3D, y, classes, groups)
    base = log_loss(y, np.tile(
        df.action.value_counts(normalize=True)[classes].values, (n, 1)), labels=classes)
    print(f"\n── 1. Out-of-fold log-loss (5-fold by game) ──────────────────────")
    print(f"  intercept-only : {base:.4f}")
    print(f"  Action-2D      : {ll2:.4f}   Δ={ll2-base:+.4f} vs intercept")
    print(f"  Action-3D      : {ll3:.4f}   Δ={ll3-base:+.4f} vs intercept  |  Δ={ll3-ll2:+.4f} vs 2D")

    # ── 2. LRT: Action-3D vs Action-2D ─────────────────────────────────────
    m2    = mnlogit(X2z, yc)
    m3    = mnlogit(X3z, yc)
    df_j  = len(POSE) * (K - 1)
    lrt_p = stats.chi2.sf(2 * (m3.llf - m2.llf), df_j)
    print(f"\n── 2. LRT: Action-3D vs Action-2D ───────────────────────────────")
    print(f"  logL(2D) = {m2.llf:.2f}  |  logL(3D) = {m3.llf:.2f}")
    print(f"  LR stat  = {2*(m3.llf-m2.llf):.2f}  df = {df_j}  p = {lrt_p:.2e}")
    verdict = "3D pose adds significant joint signal beyond 2D" if lrt_p < 0.05 \
              else "no significant joint improvement at p<0.05"
    print(f"  Verdict  : {verdict}")

    # ── 3. Action-3D coefficients ───────────────────────────────────────────
    full3 = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800).fit(X3z, y)
    cls   = list(full3.classes_); ref_i = cls.index(REF)
    rows3 = []
    for ci, c in enumerate(cls):
        if c == REF: continue
        for fi, f in enumerate(F3D):
            beta = full3.coef_[ci, fi] - full3.coef_[ref_i, fi]
            rows3.append({"action_vs_pass_kickout": c, "feature": f,
                          "log_odds_per_SD": round(beta, 4),
                          "odds_ratio_per_SD": round(np.exp(beta), 4)})
    c3 = pd.DataFrame(rows3)
    c3.to_csv("data/outputs/action_probability_3d.csv", index=False)

    piv = c3.pivot(index="feature", columns="action_vs_pass_kickout", values="odds_ratio_per_SD")
    print(f"\n── 3. Action-3D odds ratios per +1 SD (vs pass/kick-out) ────────")
    print(piv.round(3).to_string())

    # Action-2D coefficients
    full2 = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800).fit(X2z, y)
    cls2  = list(full2.classes_); r2 = cls2.index(REF)
    rows2 = [{"action_vs_pass_kickout": c, "feature": f,
               "odds_ratio_per_SD": round(np.exp(full2.coef_[ci, fi] - full2.coef_[r2, fi]), 4)}
             for ci, c in enumerate(cls2) if c != REF for fi, f in enumerate(F2D)]
    pd.DataFrame(rows2).to_csv("data/outputs/action_probability_2d.csv", index=False)

    # ── 4. Per-pose-feature independent LRT (drop-one from Action-3D) ──────
    print(f"\n── 4. Per-pose-feature independent LRT (drop-one from Action-3D) ─")
    print(f"  {'feature':<28}  {'LR stat':>8}  {'df':>4}  {'p':>10}  sig")
    per_feat_rows = []
    for pf in POSE:
        feats_drop = [f for f in F3D if f != pf]
        Xd  = StandardScaler().fit_transform(df[feats_drop])
        md  = mnlogit(Xd, yc)
        lr  = 2 * (m3.llf - md.llf)
        dfp = K - 1
        pv  = stats.chi2.sf(lr, dfp)
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {pf:<28}  {lr:>8.2f}  {dfp:>4}  {pv:>10.2e}  {sig}")
        per_feat_rows.append({"pose_feature": pf, "lr_stat": round(lr, 3),
                               "df": dfp, "p_value": round(pv, 6)})
    pd.DataFrame(per_feat_rows).to_csv(
        "data/outputs/action_probability_per_feature_lrt.csv", index=False)

    # ── 5. Predicted action distribution at three tightness levels ─────────
    # Tightness is now expressed via min_def_joint_dist_ft (3D joint distance).
    # Joint distance is typically ~1-2 ft less than centroid distance, so we
    # shift the scenario values: tight ≈ 3 ft, medium ≈ 5 ft, open ≈ 8 ft.
    means3 = df[F3D].mean(); pred = []
    for lab, d in [("tight (p10, ~4 ft)", 4.4), ("medium (p50, ~10 ft)", 9.6), ("open (p90, ~18 ft)", 17.9)]:
        row = means3.copy(); row["min_def_joint_dist_ft"] = d
        p   = full3.predict_proba(sc3.transform(row.values.reshape(1, -1)))[0]
        rec = {cls[i]: round(p[i], 4) for i in range(len(cls))}
        rec["tightness"] = lab
        rec["shot_rate"] = round(sum(rec.get(a, 0) for a in SHOTS), 4)
        pred.append(rec)
    pred_df = pd.DataFrame(pred).set_index("tightness")
    pred_df.to_csv("data/outputs/action_probability_predictions.csv")
    col_order = [c for c in ["catch_and_shoot", "contested_shot", "shot_rate",
                              "drive", "foul_turnover", "pass_kickout"]
                 if c in pred_df.columns]
    print(f"\n── 5. Predicted action distribution by closeout tightness ────────")
    print(pred_df[col_order].to_string())

    print(f"\nWrote action_probability_2d.csv, action_probability_3d.csv,")
    print(f"      action_probability_predictions.csv, action_probability_per_feature_lrt.csv")


if __name__ == "__main__":
    main()
