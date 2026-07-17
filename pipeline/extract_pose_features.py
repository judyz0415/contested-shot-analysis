#!/usr/bin/env python3
"""
extract_pose_features.py  --  Spec section 4.1 (Chapter 1)

Extract ALL 3D-skeleton-derived contest features for the nearest defender on
every shot in the 826-shot labeled sample (opponent 3PA + Heat own 3PA). One
parquet at a time. The starter list is a floor; creative pose features
(footwork, gaze, rotation, deceleration, contest timing, two-hand contest) are
implemented and tested alongside.

Output: data/intermediate/pose_features.csv  (one row per shot + shot_made)
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/OneDrive-SharedLibraries-"
    "MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025")
PRE_SHOT_OFFSET = 120; VEL_WIN = 15; DECEL_WIN = 30; CONTEST_FT = 8.0
JOINTS = ["lWrist", "rWrist", "lElbow", "rElbow", "lShoulder", "rShoulder",
          "lHip", "rHip", "lAnkle", "rAnkle", "neck", "nose"]
COLS = (["frame", "player_id", "team_id", "centroid_x", "centroid_y", "centroid_z",
         "ball_x", "ball_y", "ball_z"] + [f"{j}_{a}" for j in JOINTS for a in "xyz"])


def d2(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def d3(a, b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
def vang(dx, dy): return math.degrees(math.atan2(dy, dx))
def amid(a, b): return ((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2)


def angle_from_vertical(lower, upper):
    dx, dy, dz = upper[0]-lower[0], upper[1]-lower[1], upper[2]-lower[2]
    horiz = math.hypot(dx, dy)
    return math.degrees(math.atan2(horiz, abs(dz)+1e-9))


def angle_between(a1, a2):
    d = abs(a1 - a2) % 360
    return d if d <= 180 else 360 - d


def load_shots():
    import pandas as pd
    o = pd.read_csv("data/intermediate/shot_contest_dataset.csv")
    o = o[o.pbp_shot_result.astype(str).str.contains("made|miss", case=False)].copy()
    o["shot_made"] = o.pbp_shot_result.str.contains("made", case=False).astype(int)
    o = o[["game_file", "release_frame", "shooter_id", "nearest_defender_id", "rim_x", "rim_y",
           "closeout_speed_ft_s", "shot_made"]]
    h = pd.read_csv("data/intermediate/heat_3pa_pbp_labeled.csv").dropna(subset=["gt_made"])
    h["shot_made"] = h.gt_made.astype(int)
    h = h[["game_file", "release_frame", "shooter_id", "nearest_defender_id", "rim_x", "rim_y",
           "closeout_speed_ft_s", "shot_made"]]
    return pd.concat([o, h], ignore_index=True)


def features(rm, pteam, rf, sid, did, rimx, rimy, closeout_speed):
    """rm: (frame,pid)->row dict."""
    def g(frame, pid): return rm.get((frame, pid))
    drow = g(rf, did)
    if drow is None:
        return None
    f: Dict[str, float] = {}
    ball = (drow["ball_x"], drow["ball_y"], drow["ball_z"])

    def jt(row, name): return (row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"])
    lw, rw = jt(drow, "lWrist"), jt(drow, "rWrist")
    active = lw if lw[2] >= rw[2] else rw          # higher hand = contesting hand
    aelb = jt(drow, "lElbow") if lw[2] >= rw[2] else jt(drow, "rElbow")
    asho = jt(drow, "lShoulder") if lw[2] >= rw[2] else jt(drow, "rShoulder")

    # --- spatial contest ---
    f["true_3d_contest_distance_in"] = d3(active, ball)
    f["contest_hand_height_in"] = active[2]
    f["hand_above_release_in"] = active[2] - ball[2]
    se = d3(asho, aelb) + d3(aelb, active)
    f["arm_extension_ratio"] = d3(asho, active) / se if se > 1e-6 else float("nan")
    f["wrist_separation_in"] = d3(lw, rw)           # two-hand vs one-hand contest
    # --- balance / technique ---
    hipm = amid(jt(drow, "lHip"), jt(drow, "rHip")); shom = amid(jt(drow, "lShoulder"), jt(drow, "rShoulder"))
    f["trunk_lean_deg"] = angle_from_vertical(hipm, shom)
    la, ra = jt(drow, "lAnkle"), jt(drow, "rAnkle")
    f["foot_width_in"] = d2(la, ra)
    feet_ang = vang(ra[0]-la[0], ra[1]-la[1])
    sh_dir = vang(g(rf, sid)["centroid_x"]-drow["centroid_x"], g(rf, sid)["centroid_y"]-drow["centroid_y"]) if g(rf, sid) else feet_ang
    f["foot_alignment_deg"] = angle_between(feet_ang, sh_dir + 90)   # 0 = stance perpendicular to shooter
    hip_ang = vang(jt(drow, "rHip")[0]-jt(drow, "lHip")[0], jt(drow, "rHip")[1]-jt(drow, "lHip")[1])
    sho_ang = vang(jt(drow, "rShoulder")[0]-jt(drow, "lShoulder")[0], jt(drow, "rShoulder")[1]-jt(drow, "lShoulder")[1])
    f["hip_shoulder_rotation_deg"] = angle_between(hip_ang, sho_ang)  # 0 = squared up
    # gaze proxy: head-facing (neck->nose) vs neck->ball, in xy
    nk, no = jt(drow, "neck"), jt(drow, "nose")
    gaze = vang(no[0]-nk[0], no[1]-nk[1]); to_ball = vang(ball[0]-nk[0], ball[1]-nk[1])
    f["gaze_to_ball_deg"] = angle_between(gaze, to_ball)             # low = looking toward ball

    # --- temporal ---
    sp = g(rf - PRE_SHOT_OFFSET, sid); dp = g(rf - PRE_SHOT_OFFSET, did)
    if sp and dp:
        f["pre_shot_distance_ft"] = d2((dp["centroid_x"], dp["centroid_y"]), (sp["centroid_x"], sp["centroid_y"])) / 12
    else:
        f["pre_shot_distance_ft"] = float("nan")
    if math.isfinite(f["pre_shot_distance_ft"]) and math.isfinite(closeout_speed):
        f["reactive_speed_index"] = closeout_speed / max(f["pre_shot_distance_ft"], 0.5)
    else:
        f["reactive_speed_index"] = float("nan")
    d15 = g(rf - VEL_WIN, did)
    if d15:
        f["defender_lateral_speed_ft_s"] = d2((drow["centroid_x"], drow["centroid_y"]),
                                              (d15["centroid_x"], d15["centroid_y"])) / 12 / (VEL_WIN/60)
        f["defender_jump_in"] = drow["centroid_z"] - d15["centroid_z"]
    else:
        f["defender_lateral_speed_ft_s"] = float("nan"); f["defender_jump_in"] = float("nan")
    d30 = g(rf - DECEL_WIN, did)
    if d30 and d15:
        v_early = d2((d15["centroid_x"], d15["centroid_y"]), (d30["centroid_x"], d30["centroid_y"]))/12/(VEL_WIN/60)
        v_late = f["defender_lateral_speed_ft_s"]
        f["deceleration_ft_s"] = v_early - v_late      # positive = slowing into contest (control)
    else:
        f["deceleration_ft_s"] = float("nan")
    # contest timing: frames before release defender first got within 8 ft (in last 1s)
    ct = 0
    for k in range(1, 61):
        r = g(rf - k, did); s = g(rf - k, sid)
        if r and s and d2((r["centroid_x"], r["centroid_y"]), (s["centroid_x"], s["centroid_y"]))/12 <= CONTEST_FT:
            ct = k
    f["contest_timing_frames"] = ct                     # higher = arrived earlier

    # --- spatial context: other offensive players vs defenders ---
    steam = pteam.get(sid)
    off = [p for p, t in pteam.items() if t == steam and p != sid]
    deff = [p for p, t in pteam.items() if t != steam]
    dpos = [(g(rf, d)["centroid_x"], g(rf, d)["centroid_y"]) for d in deff if g(rf, d)]
    gaps = []
    for op in off:
        o = g(rf, op)
        if o and dpos:
            gaps.append(min(d2((o["centroid_x"], o["centroid_y"]), dp) for dp in dpos) / 12)
    if gaps:
        f["min_kickout_defender_gap_ft"] = min(gaps); f["n_open_3pt_threats"] = sum(1 for x in gaps if x > 5)
        f["avg_off_defender_gap_ft"] = sum(gaps)/len(gaps)
    else:
        f["min_kickout_defender_gap_ft"] = float("nan"); f["n_open_3pt_threats"] = 0; f["avg_off_defender_gap_ft"] = float("nan")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default="data/intermediate/pose_features.csv")
    args = ap.parse_args()
    shots = load_shots()
    print(f"Combined labeled sample: {len(shots)} shots, {shots.game_file.nunique()} games")

    rows = []; fkeys = None
    for gf, g in shots.groupby("game_file"):
        path = os.path.join(args.input_dir, gf)
        if not os.path.exists(path):
            print(f"  missing {gf}"); continue
        lo = int(g.release_frame.min()) - PRE_SHOT_OFFSET - 2
        hi = int(g.release_frame.max()) + 2
        t = pq.read_table(path, columns=COLS)
        t = t.filter(pc.and_(pc.greater_equal(t["frame"], lo), pc.less_equal(t["frame"], hi))).to_pandas()
        rm = {(int(r.frame), int(r.player_id)): r._asdict() for r in t.itertuples(index=False)}
        pteam = t.groupby("player_id").team_id.first().to_dict()
        pteam = {int(k): int(v) for k, v in pteam.items()}
        n_ok = 0
        for r in g.itertuples(index=False):
            f = features(rm, pteam, int(r.release_frame), int(r.shooter_id), int(r.nearest_defender_id),
                         r.rim_x, r.rim_y, r.closeout_speed_ft_s)
            rec = {"game_file": gf, "release_frame": int(r.release_frame), "shooter_id": int(r.shooter_id),
                   "nearest_defender_id": int(r.nearest_defender_id), "shot_made": int(r.shot_made)}
            if f:
                rec.update({k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else
                                (v if not isinstance(v, float) else "")) for k, v in f.items()})
                fkeys = fkeys or list(f.keys()); n_ok += 1
            rows.append(rec)
        print(f"  {gf}: {n_ok}/{len(g)} shots posed")
        del rm, t

    cols = ["game_file", "release_frame", "shooter_id", "nearest_defender_id"] + (fkeys or []) + ["shot_made"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nWrote {len(rows)} shots x {len(fkeys)} pose features to {args.output_csv}")
    print("features:", fkeys)


if __name__ == "__main__":
    main()
