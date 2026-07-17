#!/usr/bin/env python3
"""
sequence_action_model.py

Tests whether the TRAJECTORY of defender body state over the ~2 seconds before
the catch improves action-probability prediction beyond a single-frame snapshot.

For each perimeter touch we extract 5 pose features at N_STEPS evenly-spaced
frames ending at the catch (step size = STRIDE frames at 60 Hz):

  hand_height_in        max defender wrist z at that frame
  arm_extension_ratio   straight-line / segment-sum of active arm
  trunk_lean_deg        torso lean from vertical (shoulder midpt vs hip midpt)
  lateral_vel_ft_s      defender midHip lateral speed (centroid, 2-frame diff)
  min_joint_dist_ft     closest defender joint to handler hip-centre (3-D, feet)

Default window: STRIDE=10, N_STEPS=12  →  t = 0, -10, -20, ..., -110 frames
  = 0 to -1.83 s before the catch.
Step t=0 is the catch frame (same as the existing snapshot model).

Models compared
---------------
Snapshot-MNL   multinomial logistic on t=0 only  (replicates existing model)
Sequence-MNL   multinomial logistic on all N_STEPS × 5 features flattened
Sequence-GBM   HistGradientBoosting on flattened sequence (handles NaN natively)

Output
------
  data/outputs/sequence_logloss.csv          OOF log-loss comparison
  data/outputs/sequence_lrt.csv              LRT: sequence vs snapshot MNL
  data/outputs/sequence_step_ablation.csv    log-loss adding one time-step at a time
  data/outputs/sequence_gbm_importance.csv   GBM feature × step importance
  data/intermediate/defender_sequences.csv   cached raw sequences (re-used on re-runs)
"""
from __future__ import annotations
import argparse, math, os, sys, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
PARQUET_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)
TOUCH_CSV    = "data/intermediate/touch_events_combined.csv"
SEQ_CACHE    = "data/intermediate/defender_sequences.csv"
N_STEPS      = 12     # time steps per sequence
STRIDE       = 10     # frames between steps  (10 / 60 Hz ≈ 0.17 s)
REF_ACTION   = "pass_kickout"
CV_FOLDS     = 5

JOINTS = ["lWrist", "rWrist", "lElbow", "rElbow", "lShoulder", "rShoulder",
          "lHip", "rHip", "midHip", "lKnee", "rKnee", "lAnkle", "rAnkle"]
JCOLS  = [f"{j}_{a}" for j in JOINTS for a in "xyz"]

SEQ_FEATS = ["hand_height_in", "arm_extension_ratio", "trunk_lean_deg",
             "lateral_vel_ft_s", "min_joint_dist_ft"]
# ──────────────────────────────────────────────────────────────────────────────


# ── Feature helpers ────────────────────────────────────────────────────────────

def _pose(jrow: Optional[dict]) -> dict:
    """Defender skeleton pose features from one joint row (returns NaN dict if missing)."""
    nan = {f: float("nan") for f in SEQ_FEATS}
    if jrow is None:
        return nan
    try:
        lw = (float(jrow.get("lWrist_x") or "nan"),
              float(jrow.get("lWrist_y") or "nan"),
              float(jrow.get("lWrist_z") or "nan"))
        rw = (float(jrow.get("rWrist_x") or "nan"),
              float(jrow.get("rWrist_y") or "nan"),
              float(jrow.get("rWrist_z") or "nan"))
        hand_h = max(lw[2], rw[2])
        pfx = "l" if lw[2] >= rw[2] else "r"
        w   = lw if pfx == "l" else rw
        sh  = (float(jrow.get(f"{pfx}Shoulder_x") or "nan"),
               float(jrow.get(f"{pfx}Shoulder_y") or "nan"),
               float(jrow.get(f"{pfx}Shoulder_z") or "nan"))
        el  = (float(jrow.get(f"{pfx}Elbow_x") or "nan"),
               float(jrow.get(f"{pfx}Elbow_y") or "nan"),
               float(jrow.get(f"{pfx}Elbow_z") or "nan"))
        seg = math.dist(sh, el) + math.dist(el, w)
        ext = math.dist(sh, w) / seg if seg > 1e-6 else float("nan")

        lhip = (float(jrow.get("lHip_x") or 0), float(jrow.get("lHip_y") or 0),
                float(jrow.get("lHip_z") or 0))
        rhip = (float(jrow.get("rHip_x") or 0), float(jrow.get("rHip_y") or 0),
                float(jrow.get("rHip_z") or 0))
        hipm = ((lhip[0]+rhip[0])/2, (lhip[1]+rhip[1])/2, (lhip[2]+rhip[2])/2)

        lsho = (float(jrow.get("lShoulder_x") or 0), float(jrow.get("lShoulder_y") or 0),
                float(jrow.get("lShoulder_z") or 0))
        rsho = (float(jrow.get("rShoulder_x") or 0), float(jrow.get("rShoulder_y") or 0),
                float(jrow.get("rShoulder_z") or 0))
        shom = ((lsho[0]+rsho[0])/2, (lsho[1]+rsho[1])/2, (lsho[2]+rsho[2])/2)

        horiz = math.hypot(shom[0]-hipm[0], shom[1]-hipm[1])
        trunk = math.degrees(math.atan2(horiz, abs(shom[2]-hipm[2]) + 1e-9))
    except (TypeError, ValueError):
        return nan
    return {
        "hand_height_in":      round(hand_h, 2),
        "arm_extension_ratio": round(ext, 4),
        "trunk_lean_deg":      round(trunk, 2),
        "lateral_vel_ft_s":    float("nan"),   # filled below
        "min_joint_dist_ft":   float("nan"),   # filled below
    }


def _min_joint_dist(drow: Optional[dict], hrow: Optional[dict]) -> float:
    """Minimum 3-D distance (ft) from any defender joint to handler hip-centre."""
    if drow is None or hrow is None:
        return float("nan")
    hx = ((hrow.get("lHip_x") or 0) + (hrow.get("rHip_x") or 0)) / 2
    hy = ((hrow.get("lHip_y") or 0) + (hrow.get("rHip_y") or 0)) / 2
    hz = ((hrow.get("lHip_z") or 0) + (hrow.get("rHip_z") or 0)) / 2
    min_d = float("inf")
    for j in JOINTS:
        try:
            dx = float(drow[f"{j}_x"]); dy = float(drow[f"{j}_y"]); dz = float(drow[f"{j}_z"])
            d = math.dist((dx, dy, dz), (hx, hy, hz))
            if d < min_d:
                min_d = d
        except (KeyError, TypeError, ValueError):
            continue
    return min_d / 12.0 if min_d < float("inf") else float("nan")


# ── Sequence extraction ────────────────────────────────────────────────────────

def extract_sequences(touch_csv: str, parquet_dir: str,
                      n_steps: int, stride: int) -> pd.DataFrame:
    """
    For every touch in touch_csv, extract n_steps × len(SEQ_FEATS) values.
    Column naming: {feat}_t{k}  where k=0 is the catch frame, k=n_steps-1
    is the most-distant past frame (k * stride frames before catch).
    """
    df = pd.read_csv(touch_csv)
    valid_actions = {"catch_and_shoot", "contested_shot", "pass_kickout",
                     "drive", "foul_turnover"}
    df = df[df.action.isin(valid_actions)].reset_index(drop=True)
    print(f"Touches to sequence: {len(df):,}")

    buf = (n_steps - 1) * stride + 5
    all_rows: List[dict] = []

    for gf, grp in df.groupby("game_file"):
        path = os.path.join(parquet_dir, str(gf))
        if not os.path.exists(path):
            print(f"  [skip] {gf}"); continue

        catch_frames = grp.catch_frame.astype(int).values
        lo = int(catch_frames.min()) - buf - 3
        hi = int(catch_frames.max()) + 3

        schema_names = set(pq.read_schema(path).names)
        load_cols = ["frame", "player_id", "centroid_x", "centroid_y"] + \
                    [c for c in JCOLS if c in schema_names]

        t = pq.read_table(path, columns=load_cols)
        t = t.filter(
            pc.and_(pc.greater_equal(t["frame"], lo), pc.less_equal(t["frame"], hi))
        ).to_pandas()

        jmap: Dict[Tuple[int, int], dict] = {}
        pos:  Dict[Tuple[int, int], Tuple[float, float]] = {}
        for r in t.itertuples(index=False):
            key = (int(r.frame), int(r.player_id))
            jmap[key] = {c: getattr(r, c) for c in load_cols if c not in
                         ("frame", "player_id", "centroid_x", "centroid_y")}
            pos[key]  = (float(r.centroid_x), float(r.centroid_y))

        for r in grp.itertuples(index=False):
            cf  = int(r.catch_frame)
            hid = int(r.handler_id)
            did = int(r.nearest_def_id)
            rec = {"action": r.action, "game_file": gf,
                   "catch_frame": cf, "handler_id": hid, "nearest_def_id": did}

            for k in range(n_steps):
                frame = cf - k * stride   # t0=catch, t1=1 step back, ...
                drow  = jmap.get((frame, did))
                hrow  = jmap.get((frame, hid))
                feat  = _pose(drow)

                # lateral speed: midHip displacement over 2 frames
                p1 = pos.get((frame, did))
                p0 = pos.get((frame - 2, did))
                if p1 and p0:
                    feat["lateral_vel_ft_s"] = round(
                        math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / 12.0 / (2/60.0), 3)

                # 3-D minimum joint distance
                feat["min_joint_dist_ft"] = round(_min_joint_dist(drow, hrow), 3) \
                    if not math.isnan(_min_joint_dist(drow, hrow)
                                      if (drow is not None and hrow is not None)
                                      else float("nan")) \
                    else float("nan")

                for f, v in feat.items():
                    rec[f"{f}_t{k}"] = v

            all_rows.append(rec)

        print(f"  {gf}: {len(grp)} touches")
        del t, jmap, pos

    out = pd.DataFrame(all_rows)
    print(f"\nExtracted {len(out):,} touch sequences  "
          f"({n_steps} steps × {len(SEQ_FEATS)} features = "
          f"{n_steps * len(SEQ_FEATS)} columns per touch)\n")
    return out


# ── Model helpers ──────────────────────────────────────────────────────────────

def _oof_logloss(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 classes: list, model) -> float:
    """Out-of-fold log-loss via GroupKFold."""
    oof = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(CV_FOLDS).split(X, y, groups):
        m = model.__class__(**model.get_params()).fit(X[tr], y[tr])
        idx = [list(m.classes_).index(c) for c in classes]
        oof[te] = m.predict_proba(X[te])[:, idx]
    return log_loss(y, oof, labels=classes)


def _mnlogit_llf(X: np.ndarray, yc: np.ndarray) -> float:
    """Fit statsmodels MNLogit and return log-likelihood."""
    m = sm.MNLogit(yc, sm.add_constant(X)).fit(disp=0, maxiter=400)
    return m.llf


def seq_cols(feats: list, steps: list) -> list:
    """Return column names for given features and step indices."""
    return [f"{f}_t{k}" for k in steps for f in feats]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches",     default=TOUCH_CSV)
    ap.add_argument("--parquet-dir", default=PARQUET_DIR)
    ap.add_argument("--seq-cache",   default=SEQ_CACHE)
    ap.add_argument("--force",       action="store_true",
                    help="Re-extract sequences even if cache exists")
    ap.add_argument("--n-steps",  type=int, default=N_STEPS)
    ap.add_argument("--stride",   type=int, default=STRIDE)
    args = ap.parse_args()

    os.makedirs("data/outputs", exist_ok=True)
    os.makedirs("data/intermediate", exist_ok=True)

    # ── 1. Load or extract sequences ──────────────────────────────────────────
    if os.path.exists(args.seq_cache) and not args.force:
        print(f"Loading cached sequences from {args.seq_cache}")
        seq = pd.read_csv(args.seq_cache)
    else:
        print("Extracting sequences from parquets …")
        seq = extract_sequences(args.touches, args.parquet_dir,
                                args.n_steps, args.stride)
        seq.to_csv(args.seq_cache, index=False)
        print(f"Saved to {args.seq_cache}")

    # filter to actions with coverage in both snapshot and sequence
    valid = {"catch_and_shoot", "contested_shot", "pass_kickout", "drive", "foul_turnover"}
    seq   = seq[seq.action.isin(valid)].dropna(
        subset=[f"{f}_t0" for f in SEQ_FEATS]  # must have at least catch-frame features
    ).reset_index(drop=True)

    classes = sorted(seq.action.unique())
    K       = len(classes)
    print(f"\nn = {len(seq):,}  |  {seq.game_file.nunique()} games  |  {K} action classes")
    print("Action distribution:", seq.action.value_counts().to_dict())

    y      = seq.action.values
    groups = seq.game_file.values
    ycat   = pd.Categorical(seq.action,
                            categories=[REF_ACTION] + [c for c in classes if c != REF_ACTION])
    yc     = ycat.codes

    n_steps = args.n_steps

    # ── 2. Build X matrices ───────────────────────────────────────────────────
    snap_cols = [f"{f}_t0" for f in SEQ_FEATS]
    all_cols  = [f"{f}_t{k}" for k in range(n_steps) for f in SEQ_FEATS]

    imp  = SimpleImputer(strategy="mean")
    sc_s = StandardScaler()
    sc_a = StandardScaler()

    X_snap_raw = seq[snap_cols].values.astype(float)
    X_all_raw  = seq[all_cols].values.astype(float)

    X_snap = sc_s.fit_transform(imp.fit_transform(X_snap_raw))
    X_all  = sc_a.fit_transform(imp.fit_transform(X_all_raw))

    base_ll = log_loss(y, np.tile(
        pd.Series(y).value_counts(normalize=True).reindex(classes).values, (len(y), 1)),
        labels=classes)

    mnl  = LogisticRegression(solver="lbfgs", C=1.0, max_iter=800)
    gbm  = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, random_state=42)

    print(f"\n── 1. Out-of-fold log-loss ({CV_FOLDS}-fold by game) ──────────────────────")
    ll_base = base_ll
    ll_snap = _oof_logloss(X_snap, y, groups, classes, mnl)
    ll_seq  = _oof_logloss(X_all,  y, groups, classes, mnl)
    ll_gbm  = _oof_logloss(
        seq[all_cols].values.astype(float), y, groups, classes, gbm)

    print(f"  intercept-only  : {ll_base:.4f}")
    print(f"  Snapshot-MNL    : {ll_snap:.4f}   Δ={ll_snap-ll_base:+.4f} vs intercept")
    print(f"  Sequence-MNL    : {ll_seq:.4f}   Δ={ll_seq-ll_snap:+.4f} vs snapshot")
    print(f"  Sequence-GBM    : {ll_gbm:.4f}   Δ={ll_gbm-ll_snap:+.4f} vs snapshot")

    pd.DataFrame([
        {"model": "intercept-only", "oof_logloss": round(ll_base, 4)},
        {"model": "Snapshot-MNL",   "oof_logloss": round(ll_snap, 4)},
        {"model": "Sequence-MNL",   "oof_logloss": round(ll_seq, 4)},
        {"model": "Sequence-GBM",   "oof_logloss": round(ll_gbm, 4)},
    ]).to_csv("data/outputs/sequence_logloss.csv", index=False)

    # ── 3. LRT: Sequence-MNL vs Snapshot-MNL ─────────────────────────────────
    print(f"\n── 2. LRT: Sequence-MNL vs Snapshot-MNL ─────────────────────────────")
    llf_snap = _mnlogit_llf(X_snap, yc)
    llf_seq  = _mnlogit_llf(X_all,  yc)
    df_diff  = (len(all_cols) - len(snap_cols)) * (K - 1)
    lr_stat  = 2 * (llf_seq - llf_snap)
    lrt_p    = stats.chi2.sf(lr_stat, df_diff)
    verdict  = ("trajectory adds significant signal beyond snapshot"
                if lrt_p < 0.05 else "no significant improvement at p<0.05")
    print(f"  logL(snapshot) = {llf_snap:.2f}  |  logL(sequence) = {llf_seq:.2f}")
    print(f"  LR stat = {lr_stat:.2f}   df = {df_diff}   p = {lrt_p:.2e}")
    print(f"  Verdict: {verdict}")

    pd.DataFrame([{
        "logL_snapshot": round(llf_snap, 2), "logL_sequence": round(llf_seq, 2),
        "LR_stat": round(lr_stat, 2), "df": df_diff,
        "p_value": round(lrt_p, 6), "verdict": verdict,
    }]).to_csv("data/outputs/sequence_lrt.csv", index=False)

    # ── 4. Step ablation: how far back in time does history help? ─────────────
    print(f"\n── 3. Step ablation (adding one step at a time, earliest→catch) ─────")
    print(f"  {'Steps included':<40}  {'Window (s)':>10}  {'OOF log-loss':>12}  {'Δ vs snap':>10}")
    print("  " + "-" * 78)
    ablation_rows = []
    ll_prev = ll_snap
    for k_max in range(0, n_steps):
        cols_k = [f"{f}_t{k}" for k in range(k_max + 1) for f in SEQ_FEATS]
        Xk_raw = seq[cols_k].values.astype(float)
        Xk     = StandardScaler().fit_transform(imp.fit_transform(Xk_raw))
        ll_k   = _oof_logloss(Xk, y, groups, classes, mnl)
        window_s = k_max * args.stride / 60.0
        label    = f"t=0 to t=-{k_max * args.stride} ({window_s:.2f} s)"
        delta_snap = ll_k - ll_snap
        print(f"  {label:<40}  {window_s:>10.2f}  {ll_k:>12.4f}  {delta_snap:>+10.4f}")
        ablation_rows.append({
            "k_max": k_max, "n_steps_included": k_max + 1,
            "window_frames": k_max * args.stride,
            "window_seconds": round(window_s, 3),
            "oof_logloss": round(ll_k, 4),
            "delta_vs_snapshot": round(delta_snap, 4),
        })
        ll_prev = ll_k

    pd.DataFrame(ablation_rows).to_csv("data/outputs/sequence_step_ablation.csv", index=False)

    # ── 5. GBM feature × step importance ──────────────────────────────────────
    print(f"\n── 4. GBM feature × time-step importance ─────────────────────────────")
    from sklearn.inspection import permutation_importance
    gbm_full = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    ).fit(seq[all_cols].values.astype(float), y)

    pi = permutation_importance(gbm_full, seq[all_cols].values.astype(float), y,
                                n_repeats=5, random_state=42, n_jobs=-1)
    imp_arr = pi.importances_mean   # shape: (n_steps * n_feats,)
    # reshape to (n_steps, n_feats) then normalize per-feature
    imp_mat = imp_arr.reshape(n_steps, len(SEQ_FEATS))
    imp_df  = pd.DataFrame(imp_mat, columns=SEQ_FEATS,
                           index=[f"t-{k*args.stride}f ({k*args.stride/60:.2f}s)"
                                  for k in range(n_steps)])
    # normalise each column to sum=1 for readability
    imp_df_norm = imp_df.div(imp_df.sum(axis=0).replace(0, 1))

    print("\n  Importance share by feature and time step (columns sum to 1.0):")
    print(f"  {'Step':<28}", end="")
    for f in SEQ_FEATS:
        print(f"  {f[:18]:>18}", end="")
    print()
    print("  " + "-" * (28 + 20 * len(SEQ_FEATS)))
    for idx, row in imp_df_norm.iterrows():
        print(f"  {idx:<28}", end="")
        for v in row:
            print(f"  {v:>18.3f}", end="")
        print()

    imp_df_norm.reset_index().rename(columns={"index": "step"}).to_csv(
        "data/outputs/sequence_gbm_importance.csv", index=False)

    print(f"\nAll outputs written to data/outputs/")
    print(f"  sequence_logloss.csv, sequence_lrt.csv,")
    print(f"  sequence_step_ablation.csv, sequence_gbm_importance.csv")


if __name__ == "__main__":
    main()
