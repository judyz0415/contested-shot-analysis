#!/usr/bin/env python3
"""
feature_correlation_analysis.py  --  Spec section 4.2 (Chapter 1)

Which pose features associate with make probability, and which survive after
controlling for defender distance? Produces the correlation table, the feature
clustering matrix, and per-feature partial-r + incremental log-loss.

Input : data/intermediate/pose_features.csv (+ contest_distance_ft merged from
        the shot datasets for the distance control)
Output: data/outputs/feature_correlation_table.csv
        data/outputs/feature_correlation_matrix.csv
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

KEYS = ["game_file", "release_frame", "shooter_id", "nearest_defender_id"]


def partial_r(x, y, z):
    """partial corr of x,y controlling for z (residual method)."""
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    return stats.pearsonr(rx, ry)[0]


def main():
    pf = pd.read_csv("data/intermediate/pose_features.csv")
    # merge 2D contest distance (centroid gap) for the distance control
    cd = []
    o = pd.read_csv("data/intermediate/shot_contest_dataset.csv")[KEYS + ["contest_distance_ft"]]
    h = pd.read_csv("data/intermediate/heat_3pa_dataset.csv")[KEYS + ["contest_distance_ft"]]
    cd = pd.concat([o, h], ignore_index=True).drop_duplicates(KEYS)
    df = pf.merge(cd, on=KEYS, how="left")
    feats = [c for c in pf.columns if c not in KEYS + ["shot_made"]]
    for c in feats + ["contest_distance_ft"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["shot_made", "contest_distance_ft"]).reset_index(drop=True)
    print(f"n = {len(df)} shots, {df.game_file.nunique()} games, {len(feats)} pose features")
    y = df["shot_made"].values.astype(float)
    games = df["game_file"].values
    nfeat = len([c for c in feats if df[c].notna().sum() > 50])

    def cv_logloss(cols):
        Xs = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()))
        oof = np.zeros(len(df))
        for tr, te in GroupKFold(5).split(Xs, y, games):
            m = LogisticRegression(C=1.0, max_iter=1000).fit(Xs[tr], y[tr]); oof[te] = m.predict_proba(Xs[te])[:, 1]
        return log_loss(y, oof)
    base_ll = cv_logloss(["contest_distance_ft"])

    rows = []
    for c in feats:
        s = df[c]
        if s.notna().sum() < 50 or s.nunique() < 3:
            continue
        m = df[c].notna().values
        x = df.loc[m, c].values.astype(float); yy = y[m]; dd = df.loc[m, "contest_distance_ft"].values.astype(float)
        r, p = stats.pointbiserialr(yy, x)
        pr = partial_r(x, yy, dd)
        # incremental: LRT distance vs distance+feature, and delta CV log-loss
        d2 = df.dropna(subset=[c]).copy()
        try:
            A = sm.Logit(d2.shot_made, sm.add_constant(d2[["contest_distance_ft"]])).fit(disp=0)
            B = sm.Logit(d2.shot_made, sm.add_constant(d2[["contest_distance_ft", c]])).fit(disp=0)
            lrt_p = stats.chi2.sf(2*(B.llf - A.llf), 1)
        except Exception:
            lrt_p = float("nan")
        dll = cv_logloss(["contest_distance_ft", c]) - base_ll
        rows.append({"feature": c, "r_pointbiserial": round(r, 4), "p_value": round(p, 5),
                     "p_bonferroni": round(min(1.0, p*nfeat), 5),
                     "partial_r_ctrl_distance": round(pr, 4),
                     "lrt_p_vs_distance": round(lrt_p, 5),
                     "delta_cv_logloss_vs_distance": round(dll, 5)})
    tab = pd.DataFrame(rows).sort_values("partial_r_ctrl_distance", key=lambda s: s.abs(), ascending=False)
    tab.to_csv("data/outputs/feature_correlation_table.csv", index=False)

    # feature clustering matrix (Spearman)
    mat = df[[c for c in feats if df[c].notna().sum() > 50]].corr(method="spearman").round(3)
    mat.to_csv("data/outputs/feature_correlation_matrix.csv")

    sig = (tab.p_bonferroni < 0.05).sum()
    surv = ((tab.lrt_p_vs_distance < 0.05) & tab.lrt_p_vs_distance.notna()).sum()
    print(f"\nFeatures individually significant (Bonferroni p<0.05): {sig}/{len(tab)}")
    print(f"Features that survive controlling for distance (LRT p<0.05): {surv}/{len(tab)}")
    print(f"Baseline (distance-only) CV log-loss: {base_ll:.4f}\n")
    print(tab[["feature", "r_pointbiserial", "p_bonferroni", "partial_r_ctrl_distance",
               "lrt_p_vs_distance", "delta_cv_logloss_vs_distance"]].to_string(index=False))
    print("\nWrote data/outputs/feature_correlation_table.csv, feature_correlation_matrix.csv")


if __name__ == "__main__":
    main()
