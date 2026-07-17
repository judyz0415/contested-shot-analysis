#!/usr/bin/env python3
"""
per_defender_counterfactual.py

Per-defender counterfactual shot-rate values, restricted to Heat-defending touches.

Full Action-3D model (trained on all 6,905 pooled touches) is used to compute:
  P(shot | actual features)          — actual body-geometry at catch
  P(shot | counterfactual features)  — pose features replaced with Heat-defending means,
                                       context features kept at observed values
  delta = actual - counterfactual    — negative = defender suppresses shots below average
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HEAT_CSV     = "data/intermediate/touch_events.csv"       # Heat defending only
COMBINED_CSV = "data/intermediate/touch_events_combined.csv"
OUT_CSV      = "data/outputs/counterfactual_defender_values_heat_only.csv"
INPUT_DIR    = (
    "/Users/ruoqianzhu/Library/CloudStorage/OneDrive-SharedLibraries-"
    "MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025"
)
MIN_TOUCHES  = 30
SHOT_CLASSES = ("catch_and_shoot", "contested_shot")

F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
            "release_shot_clock", "game_clock_seconds"]
POSE    = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
           "lateral_velocity_ft_s", "min_def_joint_dist_ft"]
F3D     = F2D_CTX + POSE
REF     = "pass_kickout"


def load_and_clean(path):
    df = pd.read_csv(path)
    for c in F3D:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    for c in F3D:
        df[c] = df[c].fillna(df[c].mean())
    return df


def build_name_lookup(input_dir):
    """Scan parquets to build player_id → fullName mapping."""
    lookup = {}
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".parquet"):
            continue
        try:
            t = pq.read_table(
                os.path.join(input_dir, fname),
                columns=["player_id", "fullName"]
            ).to_pandas().drop_duplicates("player_id")
            for r in t.itertuples(index=False):
                if r.player_id not in lookup and str(r.fullName).strip():
                    lookup[int(r.player_id)] = str(r.fullName).strip()
        except Exception:
            continue
    return lookup


def predict_shot_prob(model, sc, X, classes):
    """Return P(shot) = P(CAS) + P(contested_shot) for each row in X."""
    probs = model.predict_proba(sc.transform(X))
    shot_idx = [list(model.classes_).index(c) for c in SHOT_CLASSES
                if c in model.classes_]
    return probs[:, shot_idx].sum(axis=1)


def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    combined = load_and_clean(COMBINED_CSV)
    heat_def = load_and_clean(HEAT_CSV)

    print(f"Full training set       : {len(combined)} touches")
    print(f"Heat-defending touches  : {len(heat_def)}")

    # ── 2. Fit full model on all 6,905 touches ────────────────────────────────
    classes = sorted(combined["action"].unique())
    sc_full = StandardScaler().fit(combined[F3D])
    X_train = sc_full.transform(combined[F3D])
    y_train = combined["action"].values
    model   = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800).fit(X_train, y_train)
    print(f"Model classes           : {list(model.classes_)}\n")

    # ── 3. Heat-defending pose means (the counterfactual baseline) ───────────
    pose_means_heat = heat_def[POSE].mean()
    print("Heat-defending pose feature means (counterfactual baseline):")
    for feat, val in pose_means_heat.items():
        print(f"  {feat:<35} {val:.4f}")
    print()

    # ── 4. Compute per-touch P(shot|actual) and P(shot|counterfactual) ───────
    X_actual = heat_def[F3D].values.copy()
    p_actual = predict_shot_prob(model, sc_full, X_actual, classes)

    X_cf = heat_def[F3D].values.copy()
    pose_idx = [F3D.index(f) for f in POSE]
    for pi, feat in zip(pose_idx, POSE):
        X_cf[:, pi] = pose_means_heat[feat]
    p_cf = predict_shot_prob(model, sc_full, X_cf, classes)

    heat_def = heat_def.copy()
    heat_def["p_shot_actual"]  = p_actual
    heat_def["p_shot_cf"]      = p_cf
    heat_def["shot_rate_delta"] = p_actual - p_cf   # negative = better

    # ── 5. Build player name lookup ───────────────────────────────────────────
    print("Building player name lookup from parquets …")
    name_lookup = build_name_lookup(INPUT_DIR)
    print(f"  {len(name_lookup)} players found\n")

    # ── 6. Aggregate by defender (nearest_def_id) ────────────────────────────
    agg = (
        heat_def.groupby("nearest_def_id")
        .agg(
            n_touches            = ("shot_rate_delta", "count"),
            mean_shot_actual     = ("p_shot_actual",   "mean"),
            mean_shot_cf         = ("p_shot_cf",        "mean"),
            mean_delta           = ("shot_rate_delta",  "mean"),
        )
        .reset_index()
    )

    agg = agg[agg.n_touches >= MIN_TOUCHES].copy()
    agg["pts_saved_per_touch"]   = agg["mean_delta"] * (-1) * 3 * 0.36
    agg["pts_saved_per_15games"] = agg["pts_saved_per_touch"] * agg["n_touches"]
    agg["defender_name"] = agg["nearest_def_id"].map(
        lambda pid: name_lookup.get(int(pid), f"ID_{int(pid)}")
    )

    agg = agg.sort_values("mean_delta").reset_index(drop=True)

    # reorder columns
    out = agg[["defender_name", "nearest_def_id", "n_touches",
               "mean_shot_actual", "mean_shot_cf", "mean_delta",
               "pts_saved_per_touch", "pts_saved_per_15games"]].copy()
    for col in ["mean_shot_actual", "mean_shot_cf", "mean_delta",
                "pts_saved_per_touch", "pts_saved_per_15games"]:
        out[col] = out[col].round(4)

    out.to_csv(OUT_CSV, index=False)

    # ── 7. Print table ───────────────────────────────────────────────────────
    print("── Per-defender counterfactual shot-rate delta ─────────────────────")
    print(f"  (min {MIN_TOUCHES} touches; sorted by mean_delta ascending = best → worst)\n")
    hdr = (f"  {'Defender':<22}  {'N':>5}  {'ActualSR':>9}  {'CF_SR':>7}"
           f"  {'Delta':>7}  {'PtsSaved/touch':>14}  {'PtsSaved/15g':>13}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for _, r in out.iterrows():
        mark = " ←" if r.mean_delta < -0.01 else ""
        print(
            f"  {r.defender_name:<22}  {int(r.n_touches):>5}  "
            f"{r.mean_shot_actual:>9.4f}  {r.mean_shot_cf:>7.4f}  "
            f"{r.mean_delta:>+7.4f}  {r.pts_saved_per_touch:>14.4f}  "
            f"{r.pts_saved_per_15games:>13.4f}{mark}"
        )

    print(f"\nNote: Baseline = Heat-defending pose average (not league average).")
    print(f"      Point estimates are directional; ~{int(out.n_touches.mean())}"
          f" touches/defender across 15 games.")
    print(f"\nSaved → {OUT_CSV}  ({len(out)} defenders)")


if __name__ == "__main__":
    main()
