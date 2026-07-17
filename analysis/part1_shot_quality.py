#!/usr/bin/env python3
"""
part1_v2_analysis.py  —  Part 1 Perimeter Defense (v2 Shot-Outcome Framework)

Primary dataset : shot_contest_dataset.csv (scd) merged with pose_features.csv (pf)
                  on [game_file, release_frame, nearest_defender_id]
                  → ~302–313 Heat-defending contested 3PA with pbp made/miss labels

Column mappings (printed at runtime):
  hand_above_release_in       → def_hand_elev_above_ball_in   (release-frame proxy)
  true_3d_contest_distance_in → min_def_joint_dist_ft (÷ 12 → ft)
  defender_lateral_speed_ft_s → lateral_velocity_ft_s
  arm_extension_ratio         → arm_extension_ratio       (direct)
  trunk_lean_deg              → trunk_lean_deg             (direct)
  pre_shot_distance_ft        → pre_shot_distance_ft       (direct)
  defender_jump_in (scd)      → defender_jump_in           (added in Step 3)

NOTE: these features are computed at the shot RELEASE frame, not the catch frame.
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, pearsonr, spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
SCD_CSV    = "data/intermediate/shot_contest_dataset.csv"
PF_CSV     = "data/intermediate/pose_features.csv"
TOUCH_CSV  = "data/intermediate/touch_events_combined.csv"
PLAYER_CSV = "data/PlayerStatistics.csv"
OUT_DIR    = "data/outputs/part1_v2"
FIG_DIR    = "data/outputs/part1_v2/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BAM_ID = 1628389
RNG    = np.random.default_rng(42)

# 6 core features used in Steps 1–2–4 (release-frame mapped names)
FEATS_6 = [
    "def_hand_elev_above_ball_in",   # hand_above_release_in
    "arm_extension_ratio",
    "trunk_lean_deg",
    "lateral_velocity_ft_s",         # defender_lateral_speed_ft_s
    "min_def_joint_dist_ft",         # true_3d_contest_distance_in / 12
    "pre_shot_distance_ft",
]
# Step 3 adds defender_jump_in → 7 total
FEATS_7 = FEATS_6 + ["defender_jump_in"]

BASELINE_FEATS = ["shooter_2025_26_regular_3pt_pct", "shooter_dist_to_rim_in"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _lr(X, y):
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, y)

def _loglik(m, X, y):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _bootstrap_or(X, y, n_boot=200):
    """Touch-level bootstrap → (exp_ci_lo, exp_ci_hi) arrays over features."""
    coefs = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        coefs.append(_lr(X[idx], y[idx]).coef_[0])
    coefs = np.array(coefs)
    return np.exp(np.percentile(coefs, 2.5, 0)), np.exp(np.percentile(coefs, 97.5, 0))

def _or_row(feat, coef, ci_lo, ci_hi):
    se = (np.log(ci_hi) - np.log(ci_lo)) / (2 * 1.96)
    z  = coef / (se + 1e-12)
    p  = float(2 * norm.sf(abs(z)))
    return {"feature": feat, "log_OR": round(coef, 4),
            "OR": round(np.exp(coef), 4),
            "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
            "p_value": round(p, 4)}


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("LOADING & MERGING DATA")
print("=" * 65)

scd = pd.read_csv(SCD_CSV)
pf  = pd.read_csv(PF_CSV)
ps  = pd.read_csv(PLAYER_CSV, low_memory=False)

MERGE_KEYS = ["game_file", "release_frame", "nearest_defender_id"]
raw = scd.merge(pf, on=MERGE_KEYS, how="inner", suffixes=("_scd", "_pf"))
print(f"\nMerged shape : {raw.shape}")
print(f"Columns      : {raw.columns.tolist()}")

# Eligibility filters matching v2 methodology
df = raw.copy()
for col in ("analysis_eligible", "defender_model_eligible"):
    if col in df.columns:
        df = df[df[col].astype(str).str.lower() == "yes"]
df = df[df["pbp_shot_result"].astype(str).str.contains("made|miss", case=False)].copy()
df["made"] = df["pbp_shot_result"].str.contains("made", case=False).astype(int)
df = df.reset_index(drop=True)
print(f"\nAfter eligibility + made/miss filter : {df.shape}")
print(f"Make rate: {df['made'].mean():.3f}  ({df['made'].sum()} made / {len(df)} shots)")
print(f"Defenders: {df['nearest_defender_id'].nunique()}")
print(f"Games    : {df['game_file'].nunique()}")

# ── Apply feature mappings ──────────────────────────────────────────────────
print("\nColumn mappings applied:")
df["def_hand_elev_above_ball_in"] = df["hand_above_release_in"]
print("  hand_above_release_in        → def_hand_elev_above_ball_in")
df["lateral_velocity_ft_s"] = df["defender_lateral_speed_ft_s"]
print("  defender_lateral_speed_ft_s  → lateral_velocity_ft_s")
df["min_def_joint_dist_ft"] = df["true_3d_contest_distance_in"] / 12.0
print("  true_3d_contest_distance_in / 12 → min_def_joint_dist_ft")
# defender_jump_in: prefer scd version; resolve suffix if present
if "defender_jump_in_scd" in df.columns:
    df["defender_jump_in"] = df["defender_jump_in_scd"]
    print("  defender_jump_in_scd         → defender_jump_in")
elif "defender_jump_in" not in df.columns and "defender_jump_in_pf" in df.columns:
    df["defender_jump_in"] = df["defender_jump_in_pf"]
    print("  defender_jump_in_pf          → defender_jump_in")
else:
    print("  defender_jump_in             → defender_jump_in (direct)")

print(f"\nBaseline features used for Step 0: {BASELINE_FEATS}")

# Name lookup
names = (ps[["personId","firstName","lastName"]]
         .dropna(subset=["personId"])
         .drop_duplicates("personId")
         .assign(defender_name=lambda d: d["firstName"] + " " + d["lastName"])
         .assign(personId=lambda d: d["personId"].astype(float).astype(int))
         [["personId","defender_name"]])
heat_ids = set(
    ps.loc[ps["playerteamName"] == "Heat", "personId"].dropna()
    .astype(float).astype(int)
)

# Complete-case working dataset
wdf = df[["game_file","nearest_defender_id","nearest_defender_name","made"]
         + BASELINE_FEATS + FEATS_7].dropna().copy()
print(f"\nComplete-case working dataset: {len(wdf)} rows")

y_all  = wdf["made"].values
games  = wdf["game_file"].values


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 0 — LIFT RECOMPUTATION (SHOOTER-QUALITY ADJUSTED)")
print("=" * 65)

sc_base = StandardScaler()
X_base  = sc_base.fit_transform(wdf[BASELINE_FEATS].values)
base_model = RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5,
                     fit_intercept=True, scoring="neg_log_loss")
# RidgeCV doesn't do classification; use LogisticRegressionCV for baseline
from sklearn.linear_model import LogisticRegressionCV
base_lr = LogisticRegressionCV(Cs=[100, 10, 1, 0.1], solver="lbfgs",
                                max_iter=1000, cv=5).fit(X_base, y_all)
print(f"  Baseline best C: {base_lr.C_[0]:.3f}")

wdf["p_baseline"] = base_lr.predict_proba(X_base)[:, 1]
wdf["per_shot_lift"] = (wdf["made"] - wdf["p_baseline"])

# Per-defender lift (point estimate)
lift_rows = []
for did, grp in wdf.groupby("nearest_defender_id"):
    if len(grp) < 8:
        continue
    name = grp["nearest_defender_name"].iloc[0] if "nearest_defender_name" in grp.columns else str(did)
    lift_rows.append({
        "nearest_defender_id": did,
        "defender_name": name,
        "n_shots": len(grp),
        "observed_make_rate": grp["made"].mean(),
        "lift_pp": grp["per_shot_lift"].mean() * 100,
    })
lift_df = pd.DataFrame(lift_rows)

# Game-level bootstrap CIs (B=500)
ug = wdf["game_file"].unique()
boot_lifts = {r["nearest_defender_id"]: [] for r in lift_rows}
keep_ids   = set(boot_lifts)

for _ in range(500):
    sg     = RNG.choice(ug, size=len(ug), replace=True)
    b_wdf  = pd.concat([wdf[wdf["game_file"] == g] for g in sg], ignore_index=True)
    X_b    = sc_base.transform(b_wdf[BASELINE_FEATS].values)
    p_b    = base_lr.predict_proba(X_b)[:, 1]
    b_wdf  = b_wdf.copy()
    b_wdf["per_shot_lift_b"] = (b_wdf["made"].values - p_b)
    for did in keep_ids:
        sub = b_wdf[b_wdf["nearest_defender_id"] == did]
        if len(sub) == 0:
            continue
        boot_lifts[did].append(sub["per_shot_lift_b"].mean() * 100)

lift_df["ci_lower_pp"] = lift_df["nearest_defender_id"].map(
    lambda d: float(np.percentile(boot_lifts[d], 2.5))  if boot_lifts[d] else np.nan)
lift_df["ci_upper_pp"] = lift_df["nearest_defender_id"].map(
    lambda d: float(np.percentile(boot_lifts[d], 97.5)) if boot_lifts[d] else np.nan)

lift_df = lift_df.sort_values("lift_pp").reset_index(drop=True)
out_cols = ["nearest_defender_id","defender_name","n_shots","observed_make_rate",
            "lift_pp","ci_lower_pp","ci_upper_pp"]
lift_df[out_cols].to_csv(f"{OUT_DIR}/defender_lift.csv", index=False)
print(lift_df[["defender_name","n_shots","lift_pp","ci_lower_pp","ci_upper_pp"]].to_string(index=False))
print(f"\n  Saved → {OUT_DIR}/defender_lift.csv")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 — FEATURE SUMMARY & CORRELATION HEATMAP")
print("=" * 65)

feat_df = wdf[FEATS_6].copy()
print(f"\n  {'Feature':<35s}  {'mean':>8}  {'sd':>7}  {'min':>8}  {'max':>8}")
print("  " + "-" * 72)
for f in FEATS_6:
    v = feat_df[f]
    print(f"  {f:<35s}  {v.mean():+8.3f}  {v.std():7.3f}  {v.min():+8.3f}  {v.max():+8.3f}")

corr = feat_df.corr()
print("\n  Pearson correlation matrix:")
print(corr.round(3).to_string())

print("\n  Collinear pairs (|r| > 0.7):")
flagged = []
cols = FEATS_6
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.loc[cols[i], cols[j]]
        if abs(r) > 0.7:
            print(f"    *** {cols[i]} ↔ {cols[j]}  r={r:.3f}")
            flagged.append((cols[i], cols[j], round(r, 3)))
if not flagged:
    print("    None (all |r| ≤ 0.7)")

fig, ax = plt.subplots(figsize=(8, 6))
labels = [f.replace("_", "\n") for f in FEATS_6]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            xticklabels=labels, yticklabels=labels)
ax.set_title(f"Feature Correlation Heatmap (n={len(feat_df):,} shots)", fontsize=11)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/feature_correlation_heatmap.png", dpi=150)
plt.close(fig)
print(f"\n  Saved → {FIG_DIR}/feature_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1.5 — COACHING VISUALS")
print("=" * 65)

speed_col = "lateral_velocity_ft_s"
t33, t67  = wdf[speed_col].quantile([1/3, 2/3])
scramble  = wdf[wdf[speed_col] >= t67]
stationed = wdf[wdf[speed_col] <= t33]
print(f"  Scramble (speed ≥ {t67:.2f} ft/s): n={len(scramble)}")
print(f"  Stationed (speed ≤ {t33:.2f} ft/s): n={len(stationed)}")

# KDE grid
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(FEATS_6):
    ax  = axes[i]
    sv  = scramble[feat].dropna()
    pv  = stationed[feat].dropna()
    u_stat, mw_p = mannwhitneyu(sv, pv, alternative="two-sided")
    delta = sv.mean() - pv.mean()
    for label, vals, c in [("Scramble", sv, "#C0392B"), ("Stationed", pv, "#2980B9")]:
        vals.plot.kde(ax=ax, label=label, color=c, lw=2, alpha=0.85)
        ax.axvline(vals.mean(), color=c, ls="--", lw=1, alpha=0.7)
    ax.set_title(feat.replace("_", " "), fontsize=9, fontweight="bold")
    ax.annotate(f"Δmean={delta:+.2f}\np={mw_p:.3f}",
                xy=(0.97, 0.90), xycoords="axes fraction",
                ha="right", fontsize=7.5, color="#444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))
    ax.legend(fontsize=7); ax.set_xlabel(""); ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=7)

fig.suptitle("Feature Distributions: Scramble vs Stationed Closeouts (shot level)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/feature_distributions_coaching.png", dpi=150)
plt.close(fig)
print(f"  Saved → {FIG_DIR}/feature_distributions_coaching.png")

# Distance vs speed scatter
fig, ax = plt.subplots(figsize=(8, 6))
colors_made = wdf["made"].map({1: "#C0392B", 0: "#27AE60"})
non_bam = wdf[wdf["nearest_defender_id"] != BAM_ID]
bam_df  = wdf[wdf["nearest_defender_id"] == BAM_ID]
ax.scatter(non_bam["pre_shot_distance_ft"], non_bam["lateral_velocity_ft_s"],
           c=non_bam["made"].map({1: "#C0392B", 0: "#27AE60"}),
           alpha=0.5, s=30, label=None)
ax.scatter(bam_df["pre_shot_distance_ft"], bam_df["lateral_velocity_ft_s"],
           c=bam_df["made"].map({1: "#C0392B", 0: "#27AE60"}),
           marker="D", s=70, edgecolors="black", lw=1.2, label="Bam Adebayo", zorder=5)
# legend proxies
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#C0392B", ms=8, label="Made"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#27AE60", ms=8, label="Missed"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="gray", ms=8,
           markeredgecolor="black", label="Bam Adebayo"),
]
ax.legend(handles=legend_els, fontsize=8)
ax.set_xlabel("Pre-shot distance (ft)", fontsize=10)
ax.set_ylabel("Defender lateral speed (ft/s)", fontsize=10)
ax.set_title("Faster defenders started farther away", fontsize=11, fontweight="bold")
ax.grid(True, lw=0.4, alpha=0.4)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/distance_vs_speed_scatter.png", dpi=150)
plt.close(fig)
print(f"  Saved → {FIG_DIR}/distance_vs_speed_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — LOGISTIC REGRESSION: FEATURES → MADE")
print("=" * 65)

sc6   = StandardScaler()
X6    = sc6.fit_transform(wdf[FEATS_6].values)
y     = y_all

# Univariate
print("\n  Univariate (each feature alone):")
uni_rows = []
for j, feat in enumerate(FEATS_6):
    Xj = X6[:, [j]]
    m  = _lr(Xj, y)
    c  = float(m.coef_[0, 0])
    lo, hi = _bootstrap_or(Xj, y)
    row = _or_row(feat, c, lo[0], hi[0])
    uni_rows.append(row)
    sig = " *" if row["p_value"] < 0.05 else ""
    print(f"  {feat:<35s}  OR={row['OR']:.3f} [{row['ci_lower']:.3f},{row['ci_upper']:.3f}]"
          f"  p={row['p_value']:.4f}{sig}")

uni_df = pd.DataFrame(uni_rows)
uni_df.to_csv(f"{OUT_DIR}/step2_univariate.csv", index=False)
print(f"\n  Saved → {OUT_DIR}/step2_univariate.csv")

# Multivariate
print("\n  Multivariate (all 6 features):")
m6   = _lr(X6, y)
lo6, hi6 = _bootstrap_or(X6, y)
mv_rows  = []
mv_coefs = m6.coef_[0]
for j, feat in enumerate(FEATS_6):
    row = _or_row(feat, float(mv_coefs[j]), lo6[j], hi6[j])
    mv_rows.append(row)
    sig = " *" if row["p_value"] < 0.05 else ""
    print(f"  {feat:<35s}  OR={row['OR']:.3f} [{row['ci_lower']:.3f},{row['ci_upper']:.3f}]"
          f"  p={row['p_value']:.4f}{sig}")

mv_df = pd.DataFrame(mv_rows)
mv_df.to_csv(f"{OUT_DIR}/step2_multivariate.csv", index=False)
print(f"\n  Saved → {OUT_DIR}/step2_multivariate.csv")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — CONTROL FOR 2D DISTANCE (7 features)")
print("=" * 65)

sc7  = StandardScaler()
X7   = sc7.fit_transform(wdf[FEATS_7].values)
m7   = _lr(X7, y)
lo7, hi7 = _bootstrap_or(X7, y)
ctrl_rows = []
surviving_pose = []
print(f"\n  Model with {len(FEATS_7)} features (pre_shot_distance_ft forced + 5 pose + defender_jump_in):")
for j, feat in enumerate(FEATS_7):
    row = _or_row(feat, float(m7.coef_[0, j]), lo7[j], hi7[j])
    ctrl_rows.append(row)
    marker = " ← p<0.05" if row["p_value"] < 0.05 else ""
    print(f"  {feat:<35s}  OR={row['OR']:.3f}  p={row['p_value']:.4f}{marker}")
    if row["p_value"] < 0.05 and feat != "pre_shot_distance_ft":
        surviving_pose.append(feat)

ctrl_df = pd.DataFrame(ctrl_rows)
ctrl_df.to_csv(f"{OUT_DIR}/step3_distance_controlled.csv", index=False)
print(f"\nPose features surviving 2D distance control: {surviving_pose}")
print(f"  Saved → {OUT_DIR}/step3_distance_controlled.csv")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — SCQ COMPOSITE")
print("=" * 65)

# Use Step 2 multivariate coefficients (6 features)
# SCQ = -sum(coeff_j * zscore_j)  so higher SCQ = better contest
# (equivalently: flip sign of each feature where coeff > 0)
z6 = stats.zscore(wdf[FEATS_6].values, nan_policy="omit")
scq_raw = z6 @ mv_coefs
wdf["SCQ"] = -scq_raw  # negate: higher = tighter contest = lower P(made)

print("  Feature contributions to SCQ (positive coef → negated → protective):")
for feat, c in zip(FEATS_6, mv_coefs):
    direction = "→ FLIPPED (higher value raises P(made))" if c > 0 else "→ kept    (higher value lowers P(made))"
    print(f"    {feat:<35s}  coef={c:+.4f}  {direction}")

# Aggregate to defender level
scq_grp = wdf.groupby("nearest_defender_id").agg(
    defender_name=("nearest_defender_name", "first"),
    n_shots=("made", "count"),
    mean_scq=("SCQ", "mean"),
).reset_index()

scq_df = (scq_grp[scq_grp["n_shots"] >= 8]
          .merge(lift_df[["nearest_defender_id","lift_pp","ci_lower_pp","ci_upper_pp"]],
                 on="nearest_defender_id", how="inner")
          .sort_values("mean_scq", ascending=False)
          .reset_index(drop=True))

scq_df.to_csv(f"{OUT_DIR}/step4_scq_by_defender.csv", index=False)
print("\n  SCQ by defender:")
print(scq_df[["defender_name","n_shots","mean_scq","lift_pp"]].to_string(index=False))
print(f"\n  Saved → {OUT_DIR}/step4_scq_by_defender.csv")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — SCQ vs SUPPRESSION (OLS + RIDGE + SCATTER)")
print("=" * 65)

X_scq  = scq_df["mean_scq"].values.reshape(-1, 1)
y_lift = scq_df["lift_pp"].values

ols   = LinearRegression().fit(X_scq, y_lift)
r5, p5 = pearsonr(scq_df["mean_scq"], scq_df["lift_pp"])
r2_ols = r5 ** 2
print(f"  OLS:  slope={ols.coef_[0]:.3f}  R²={r2_ols:.3f}  p={p5:.4f}")

ridge = RidgeCV(alphas=[0.1, 1, 10, 100], cv=LeaveOneOut()).fit(X_scq, y_lift)
ss_res  = np.sum((y_lift - ridge.predict(X_scq)) ** 2)
ss_tot  = np.sum((y_lift - y_lift.mean()) ** 2)
r2_ridge = 1 - ss_res / ss_tot
print(f"  Ridge: best_alpha={ridge.alpha_:.1f}  R²(train)={r2_ridge:.3f}")

x_line = np.linspace(X_scq.min() - 0.05, X_scq.max() + 0.05, 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_line, ols.predict(x_line), color="#C0392B", lw=1.8,
        label=f"OLS  R²={r2_ols:.2f}  p={p5:.3f}")
ax.plot(x_line, ridge.predict(x_line), color="#2980B9", lw=1.8, ls="--",
        label=f"Ridge α={ridge.alpha_:.0f}  R²={r2_ridge:.2f}")
for _, row in scq_df.iterrows():
    ax.errorbar(row["mean_scq"], row["lift_pp"],
                yerr=[[row["lift_pp"] - row["ci_lower_pp"]],
                      [row["ci_upper_pp"] - row["lift_pp"]]],
                fmt="o", color="#555", ms=6, lw=1, alpha=0.7, capsize=3)
    ax.annotate(row["defender_name"].split()[-1],
                xy=(row["mean_scq"], row["lift_pp"]),
                xytext=(5, 3), textcoords="offset points", fontsize=8)
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("Mean SCQ (higher = tighter contest)", fontsize=10)
ax.set_ylabel("Lift pp (vs shooter-quality baseline)", fontsize=10)
ax.set_title("SCQ vs Shot Suppression - Heat Defenders", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/scq_vs_suppression.png", dpi=150)
plt.close(fig)
print(f"  Saved → {FIG_DIR}/scq_vs_suppression.png")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — SPEED PARADOX")
print("=" * 65)

# (A) Per-touch mechanism
touch = pd.read_csv(TOUCH_CSV)
touch_heat = touch[touch["nearest_def_id"].isin(heat_ids)].dropna(
    subset=["def_dist_at_catch_ft","lateral_velocity_ft_s"])
r_touch, p_touch = pearsonr(touch_heat["def_dist_at_catch_ft"],
                             touch_heat["lateral_velocity_ft_s"])
print(f"\n  (A) Per-touch r(dist, speed) = {r_touch:.3f}  (p={p_touch:.4g}, n={len(touch_heat):,})")

# (B) Per-defender roster level from shot dataset
per_def = (wdf.groupby("nearest_defender_id")
           [["pre_shot_distance_ft","lateral_velocity_ft_s"]].mean()
           .reset_index())
per_def = per_def[per_def["nearest_defender_id"].isin(keep_ids)].dropna()
r_def, p_def = pearsonr(per_def["pre_shot_distance_ft"], per_def["lateral_velocity_ft_s"])
print(f"  (B) Per-defender r(dist, speed) = {r_def:.3f}  (p={p_def:.4f}, n={len(per_def)})")

# Sensitivity: vary speed weight
speed_idx = FEATS_6.index("lateral_velocity_ft_s")
spearman_rs = []
for mult in np.linspace(0, 3, 31):
    c_mod = mv_coefs.copy()
    c_mod[speed_idx] *= mult
    scq_mod = -(z6 @ c_mod)
    wdf_copy = wdf.copy()
    wdf_copy["SCQ_mod"] = scq_mod
    per_def_scq = (wdf_copy.groupby("nearest_defender_id")["SCQ_mod"].mean()
                   .reset_index().rename(columns={"SCQ_mod":"mean_scq_mod"}))
    merged_s = per_def_scq.merge(lift_df[["nearest_defender_id","lift_pp"]], on="nearest_defender_id")
    r_s, _ = spearmanr(merged_s["mean_scq_mod"], merged_s["lift_pp"])
    spearman_rs.append(r_s)

mults = np.linspace(0, 3, 31)
base_r6 = spearman_rs[np.argmin(np.abs(mults - 1.0))]
print(f"  Spearman r(SCQ, lift) at baseline speed weight (×1.0): {base_r6:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mults, spearman_rs, color="#2980B9", lw=2)
ax.axvline(1.0, color="#C0392B", ls="--", lw=1.5, label=f"Baseline (r={base_r6:.2f})")
ax.set_xlabel("lateral_velocity_ft_s weight multiplier", fontsize=10)
ax.set_ylabel("Spearman r (SCQ vs lift_pp)", fontsize=10)
ax.set_title("Predictive validity of SCQ degrades as speed weight increases",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, lw=0.4, alpha=0.4)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/scq_sensitivity_speed_weight.png", dpi=150)
plt.close(fig)
print(f"  Saved → {FIG_DIR}/scq_sensitivity_speed_weight.png")

# Bam profile
bam_row = scq_df[scq_df["nearest_defender_id"] == BAM_ID]
if not bam_row.empty:
    bam = bam_row.iloc[0]
    per_def_full = (wdf.groupby("nearest_defender_id")
                    [["pre_shot_distance_ft","lateral_velocity_ft_s"]].mean()
                    .reset_index())
    per_def_full["dist_rank"]  = per_def_full["pre_shot_distance_ft"].rank()
    per_def_full["speed_rank"] = per_def_full["lateral_velocity_ft_s"].rank()
    bam_stats = per_def_full[per_def_full["nearest_defender_id"] == BAM_ID].iloc[0]
    n_defs = len(per_def_full)
    print(f"\n  Bam Adebayo profile (vs {n_defs} defenders in shot dataset):")
    print(f"    mean pre_shot_distance  : {bam_stats['pre_shot_distance_ft']:.2f} ft"
          f"  (rank {int(bam_stats['dist_rank'])} / {n_defs} — lower rank = closer)")
    print(f"    mean lateral speed      : {bam_stats['lateral_velocity_ft_s']:.2f} ft/s"
          f"  (rank {int(bam_stats['speed_rank'])} / {n_defs})")
    print(f"    lift_pp                 : {bam['lift_pp']:+.2f} pp"
          f"  [{bam['ci_lower_pp']:+.1f}, {bam['ci_upper_pp']:+.1f}]")
    print(f"    mean_scq                : {bam['mean_scq']:+.3f}")
else:
    print("  Bam not in ≥8-shot defender set")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 — NOISE WALL (SHOT-OUTCOME AUC)")
print("=" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_dist = sc6.transform(wdf[["pre_shot_distance_ft"] +
                             [f for f in FEATS_6 if f != "pre_shot_distance_ft"]])[:, [5]]
# just distance column (index 5 = pre_shot_distance_ft in FEATS_6)
dist_idx = FEATS_6.index("pre_shot_distance_ft")
X_dist   = X6[:, [dist_idx]]

models_7 = {
    "dummy":          DummyClassifier(strategy="most_frequent"),
    "dist_only_LR":   LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000),
    "full_6feat_LR":  LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000),
    "full_6feat_XGB": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42, verbosity=0,
                                    eval_metric="logloss"),
}
X_map7 = {
    "dummy":          X6,
    "dist_only_LR":   X_dist,
    "full_6feat_LR":  X6,
    "full_6feat_XGB": X6,
}
auc_rows = []
for name, model in models_7.items():
    scores = cross_val_score(model, X_map7[name], y, cv=skf, scoring="roc_auc")
    auc_rows.append({"model": name, "mean_auc": round(scores.mean(), 4),
                     "std_auc": round(scores.std(), 4)})
    print(f"  {name:<22s}  AUC={scores.mean():.3f} ± {scores.std():.3f}")

# LRT: full_6feat_LR vs dist_only_LR
m_b = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X_dist, y)
m_c = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X6,     y)
ll_b = _loglik(m_b, X_dist, y)
ll_c = _loglik(m_c, X6,     y)
lrt_stat = 2 * (ll_c - ll_b)
lrt_p    = float(stats.chi2.sf(lrt_stat, df=5))
print(f"\n  LRT (full_6 vs dist_only): χ²={lrt_stat:.2f}  df=5  p={lrt_p:.4f}")

auc_df = pd.DataFrame(auc_rows)
auc_df["lrt_p_full_vs_dist"] = [None, None, lrt_p, None]
auc_df.to_csv(f"{OUT_DIR}/step7_noise_wall.csv", index=False)
print(f"  Saved → {OUT_DIR}/step7_noise_wall.csv")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

sig_uni   = uni_df[uni_df["p_value"] < 0.05]["feature"].tolist()
top_lift  = lift_df.iloc[0]
bot_lift  = lift_df.iloc[-1]

print(f"""
  Step 0 — Lift
    Grand mean (shooter-adjusted baseline): p_baseline mean = {wdf['p_baseline'].mean():.3f}
    Best suppressor  : {top_lift['defender_name']}  lift={top_lift['lift_pp']:+.1f} pp  [{top_lift['ci_lower_pp']:+.1f},{top_lift['ci_upper_pp']:+.1f}]
    Worst suppressor : {bot_lift['defender_name']}  lift={bot_lift['lift_pp']:+.1f} pp  [{bot_lift['ci_lower_pp']:+.1f},{bot_lift['ci_upper_pp']:+.1f}]
    Defenders (≥8 shots): {len(lift_df)}

  Step 1 — Collinear pairs flagged (|r|>0.7): {flagged if flagged else 'None'}

  Step 2 — Significant univariate predictors (p<0.05): {sig_uni}

  Step 3 — Pose features surviving 2D distance control: {surviving_pose}

  Step 5 — SCQ vs suppression
    OLS R²={r2_ols:.3f}  p={p5:.4f}   Ridge R²={r2_ridge:.3f}  best_alpha={ridge.alpha_:.0f}

  Step 6 — Speed paradox
    Per-touch r(dist, speed)     = {r_touch:.3f}  (n={len(touch_heat):,})
    Per-defender r(dist, speed)  = {r_def:.3f}   (n={len(per_def)})
    Spearman r(SCQ, lift) at baseline speed weight = {base_r6:.3f}

  Step 7 — Noise wall (n={len(y)} shots)
""")
for _, row in auc_df.iterrows():
    print(f"    {row['model']:<22s}  AUC={row['mean_auc']:.3f} ± {row['std_auc']:.3f}")
print(f"\n    LRT p = {lrt_p:.4f}")
