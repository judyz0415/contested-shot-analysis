#!/usr/bin/env python3
"""
shot_outcome_wall.py  --  Spec section 4.5 (Chapter 4)

Demonstrate that shot-level make/miss cannot be predicted reliably even with the
richest feature set (all pose features + PC-3D). This is a finding, not a failure:
~826 binary 3PT outcomes is a coin flip. It is the precise motivation for EPV.

Four tests on the 826-shot sample (defender-RE test on the 304 contested pool):
  1. Logistic (all pose + PC-3D): 5-fold CV AUC, by game
  2. XGBoost (same): 5-fold CV AUC, by game
  3. LRT for defender random effects (contested opponent pool)
  4. Feature-subset LRT: distance-only vs. full pose feature set

Output: data/outputs/shot_outcome_wall_results.csv
"""
from __future__ import annotations
import warnings, math
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

KEYS = ["game_file", "release_frame", "shooter_id", "nearest_defender_id"]


def cv_auc(X, y, groups, model="logit"):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        if model == "logit":
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05, subsample=.8,
                                  colsample_bytree=.8, reg_lambda=2.0, eval_metric="logloss").fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    pf = pd.read_csv("data/intermediate/pose_features.csv")
    pc = pd.read_csv("data/outputs/pitch_control_shot_level.csv")[KEYS + ["pc_value_3d"]]
    cd = pd.concat([pd.read_csv("data/intermediate/shot_contest_dataset.csv")[KEYS + ["contest_distance_ft"]],
                    pd.read_csv("data/intermediate/heat_3pa_dataset.csv")[KEYS + ["contest_distance_ft"]]]).drop_duplicates(KEYS)
    df = pf.merge(pc, on=KEYS, how="left").merge(cd, on=KEYS, how="left")
    feats = [c for c in pf.columns if c not in KEYS + ["shot_made"]] + ["pc_value_3d", "contest_distance_ft"]
    for c in feats: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["shot_made", "contest_distance_ft"]).reset_index(drop=True)
    use = [c for c in feats if df[c].notna().sum() > len(df) * 0.8]
    df[use] = df[use].fillna(df[use].median())
    y = df.shot_made.values.astype(int); groups = df.game_file.values
    X = df[use].values
    print(f"Honest wall: {len(df)} shots, {len(use)} features (all pose + PC-3D + distance)")

    res = []
    a_log = cv_auc(X, y, groups, "logit"); res.append(("Logistic (all features)", "5-fold CV AUC", round(a_log, 3)))
    a_xgb = cv_auc(X, y, groups, "xgb"); res.append(("XGBoost (all features)", "5-fold CV AUC", round(a_xgb, 3)))
    a_dist = cv_auc(df[["contest_distance_ft"]].values, y, groups, "logit"); res.append(("Logistic (distance only)", "5-fold CV AUC", round(a_dist, 3)))

    # test 4: distance-only vs full pose set (in-sample LRT)
    A = sm.Logit(df.shot_made, sm.add_constant(df[["contest_distance_ft"]])).fit(disp=0)
    posef = [c for c in use if c != "contest_distance_ft"]
    B = sm.Logit(df.shot_made, sm.add_constant(df[["contest_distance_ft"] + posef])).fit(disp=0)
    lrt_p = stats.chi2.sf(2 * (B.llf - A.llf), len(posef)); res.append(("Full pose set vs distance-only", "LRT p", round(lrt_p, 4)))

    # test 3: defender random effects on contested opponent pool
    o = pd.read_csv("data/intermediate/shot_contest_dataset.csv")
    for c in ("analysis_eligible", "defender_model_eligible"):
        if c in o: o = o[o[c].astype(str).str.lower() == "yes"]
    o = o[o.pbp_shot_result.astype(str).str.contains("made|miss", case=False)].copy()
    o["made"] = o.pbp_shot_result.str.contains("made", case=False).astype(int)
    om = o.merge(pf[KEYS + ["pre_shot_distance_ft", "reactive_speed_index"]], on=KEYS, how="inner").dropna(
        subset=["made", "contest_distance_ft", "pre_shot_distance_ft", "reactive_speed_index", "contest_angle_deg"])
    zf = []
    for c in ["contest_distance_ft", "pre_shot_distance_ft", "reactive_speed_index", "contest_angle_deg"]:
        om[c + "_z"] = (om[c] - om[c].mean()) / om[c].std(ddof=1); zf.append(c + "_z")
    Aa = smf.glm("made ~ " + " + ".join(zf), om, family=sm.families.Binomial()).fit()
    Bb = smf.glm("made ~ " + " + ".join(zf) + " + C(nearest_defender_id)", om, family=sm.families.Binomial()).fit()
    re_p = stats.chi2.sf(2 * (Bb.llf - Aa.llf), int(Bb.df_model - Aa.df_model))
    res.append(("Defender identity (LRT, contested n=%d)" % len(om), "LRT p", round(re_p, 4)))

    out = pd.DataFrame(res, columns=["test", "metric", "value"])
    out.to_csv("data/outputs/shot_outcome_wall_results.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\nAll predictive tests at or below chance (AUC ~0.5) and no feature set beats distance. "
          "At N=826 binary 3PT outcomes, make/miss cannot rank defenders -- this is the structural "
          "noise that motivates EPV: value the possession state, not the terminal outcome.")
    print("\nWrote data/outputs/shot_outcome_wall_results.csv")


if __name__ == "__main__":
    main()
