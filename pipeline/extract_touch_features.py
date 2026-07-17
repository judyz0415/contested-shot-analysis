#!/usr/bin/env python3
"""
extract_touch_features.py  --  feeds the action-probability model (spec 4.3)

The 3,570 perimeter touches in closeout_actions.csv have only the ending and the
defender distance. Spec 4.3 needs each touch enriched with closeout-state
features at the CATCH frame -- pitch control, spatial context, and clocks -- and
the SHOT ending split into catch_and_shoot vs. contested_shot. This script
computes those from the raw parquets (one game at a time) and emits a modeling-
ready touch dataset.

Action categories (spec 4.3):
  catch_and_shoot   shot released <= 0.7 s after the catch (no dribble/pump)
  contested_shot    shot released later (dribble / pump-fake / late contest)
  pass_kickout      ball swung / returned to perimeter (PASS)
  drive             ball-handler attacked the paint (DRIVE)
  foul_turnover     live-ball turnover (TURNOVER)        [fouls under-captured]
  -- OTHER touches (end-of-clock / resets) are dropped (not an offensive action)

Features (at the catch frame): def_dist_at_catch_ft (defender distance at catch),
pitch_control_value (defensive control of the handler's spot), n_open_3pt_threats,
min_kickout_defender_gap_ft, release_shot_clock, game_clock_seconds (sec left in period).

Output: data/intermediate/touch_events.csv
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)
import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/OneDrive-SharedLibraries-"
    "MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025")
VEL_WIN = 12               # 0.2 s velocity window
CS_THRESHOLD_S = 0.7       # catch-to-release <= this -> catch_and_shoot
OPEN_GAP_FT = 5.0
# pitch-control constants (literature defaults, identical to validated model)
MAX_SPEED, REACTION, SIGMA, LAM, DT, TMAX = 16.4, 0.3, 0.45, 4.3, 0.04, 6.0


def _tti(loc, p):
    rx, ry = p[0] + p[2] * REACTION, p[1] + p[3] * REACTION
    return REACTION + math.hypot(loc[0] - rx, loc[1] - ry) / MAX_SPEED


def def_control(loc, players):
    taus = [_tti(loc, p) for p in players]; isd = [p[4] for p in players]
    ppcf = [0.0] * len(players); coef = math.pi / (math.sqrt(3) * SIGMA)
    t, tot = max(0.0, min(taus) - 3 * SIGMA), 0.0
    while tot < 0.99 and t < TMAX:
        for i in range(len(players)):
            if ppcf[i] >= 1: continue
            d = (1 - tot) * (1 / (1 + math.exp(-coef * (t - taus[i])))) * LAM * DT
            ppcf[i] += d; tot += d
        t += DT
    return sum(ppcf[i] for i in range(len(players)) if isd[i])


def _clk(s) -> Optional[float]:
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    if ":" in s:
        a, b = s.split(":"); return int(a) * 60 + float(b)
    try:
        return float(s)
    except ValueError:
        return None


ACTION_MAP = {"DRIVE": "drive", "PASS": "pass_kickout", "TURNOVER": "foul_turnover"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", default="data/intermediate/closeout_actions.csv")
    ap.add_argument("--shots", default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default="data/intermediate/touch_events.csv")
    ap.add_argument("--offense", default="opponent", choices=["opponent", "heat"],
                    help="'opponent': Heat defends (default); 'heat': opponents defend Heat")
    args = ap.parse_args()
    # heat-offense mode: wire up the flipped files automatically
    if args.offense == "heat":
        if args.touches == "data/intermediate/closeout_actions.csv":
            args.touches = "data/intermediate/closeout_actions_heat_offense.csv"
        if args.shots == "data/intermediate/shot_contest_dataset.csv":
            args.shots = "data/intermediate/heat_3pa_pbp_labeled.csv"
        if args.output_csv == "data/intermediate/touch_events.csv":
            args.output_csv = "data/intermediate/touch_events_heat_offense.csv"

    import pandas as pd
    JOINTS = ["lWrist", "rWrist", "lElbow", "rElbow", "lShoulder", "rShoulder",
              "lHip", "rHip", "midHip", "lKnee", "rKnee", "lAnkle", "rAnkle"]

    def _min_joint_dist_ft(drow, hrow):
        """Min 3-D distance (ft) from any defender joint to the handler's hip-centre."""
        if drow is None or hrow is None:
            return ""
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
        return round(min_d / 12.0, 3) if min_d < float("inf") else ""

    def pose_at(jrow, ball_z=None):
        """defender pose features from a joint row dict (or None)."""
        if jrow is None:
            return {}
        def jt(n): return (jrow[f"{n}_x"], jrow[f"{n}_y"], jrow[f"{n}_z"])
        lw, rw = jt("lWrist"), jt("rWrist")
        active = "l" if lw[2] >= rw[2] else "r"
        w = lw if active == "l" else rw
        sh = jt(f"{active}Shoulder"); el = jt(f"{active}Elbow")
        se = math.dist(sh, el) + math.dist(el, w)
        hipm = [(jrow["lHip_x"] + jrow["rHip_x"]) / 2, (jrow["lHip_y"] + jrow["rHip_y"]) / 2, (jrow["lHip_z"] + jrow["rHip_z"]) / 2]
        shom = [(jrow["lShoulder_x"] + jrow["rShoulder_x"]) / 2, (jrow["lShoulder_y"] + jrow["rShoulder_y"]) / 2, (jrow["lShoulder_z"] + jrow["rShoulder_z"]) / 2]
        horiz = math.hypot(shom[0] - hipm[0], shom[1] - hipm[1])
        elev = round(w[2] - ball_z, 2) if ball_z is not None else ""
        return {"def_hand_elev_above_ball_in": elev,
                "arm_extension_ratio": round(math.dist(sh, w) / se, 4) if se > 1e-6 else "",
                "trunk_lean_deg": round(math.degrees(math.atan2(horiz, abs(shom[2] - hipm[2]) + 1e-9)), 2)}

    tdf = pd.read_csv(args.touches)
    sdf = pd.read_csv(args.shots)
    # shot release frames per game+shooter, to split catch_and_shoot vs contested
    shot_rel = defaultdict(list)
    for r in sdf.itertuples(index=False):
        shot_rel[(r.game_file, int(r.shooter_id))].append(int(r.release_frame))

    rows: List[dict] = []
    for gf, g in tdf.groupby("game_file"):
        path = os.path.join(args.input_dir, gf)
        if not os.path.exists(path):
            print(f"  missing {gf}"); continue
        jcols = [f"{j}_{a}" for j in JOINTS for a in "xyz"]
        t = pq.read_table(path, columns=["frame", "player_id", "team_id", "teamName",
                                         "centroid_x", "centroid_y", "shotClockTime",
                                         "gameClockTime", "ball_z"] + jcols).to_pandas()
        pos = {(int(r.frame), int(r.player_id)): (r.centroid_x, r.centroid_y) for r in t.itertuples(index=False)}
        ball_z_map = t.drop_duplicates("frame").set_index("frame")["ball_z"].to_dict()
        # defender pose lookup only at the catch frames we need (memory-safe)
        need = ({(int(r.catch_frame), int(r.nearest_def_id)) for r in g.itertuples(index=False)} |
                {(int(r.catch_frame), int(r.handler_id))    for r in g.itertuples(index=False)})
        jt = t[["frame", "player_id"] + jcols]
        jmap = {(int(r.frame), int(r.player_id)): {c: getattr(r, c) for c in jcols}
                for r in jt.itertuples(index=False) if (int(r.frame), int(r.player_id)) in need}
        pteam = t.groupby("player_id").team_id.first().to_dict()
        fr = t[["frame", "shotClockTime", "gameClockTime"]].drop_duplicates("frame")
        sc = {int(r.frame): _clk(r.shotClockTime) for r in fr.itertuples(index=False)}
        gc = {int(r.frame): _clk(r.gameClockTime) for r in fr.itertuples(index=False)}
        allpids = list(pteam.keys())
        n_ok = 0
        for r in g.itertuples(index=False):
            cf = int(r.catch_frame); hid = int(r.handler_id); ending = r.ending
            # action category
            if ending == "OTHER":
                continue
            if ending == "SHOT":
                rels = [f for f in shot_rel.get((gf, hid), []) if cf <= f <= cf + 40]
                if rels:
                    cr = (min(rels) - cf) / 60.0
                    action = "catch_and_shoot" if cr <= CS_THRESHOLD_S else "contested_shot"
                else:
                    action = "contested_shot"; cr = float("nan")
            else:
                action = ACTION_MAP.get(ending); cr = float("nan")
                if action is None:
                    continue
            hpos = pos.get((cf, hid))
            if hpos is None:
                continue
            hteam = pteam.get(hid)
            # players at catch with velocity over VEL_WIN
            players = []; def_pos = []; off_other = []
            for pid in allpids:
                p1 = pos.get((cf, pid)); p0 = pos.get((cf - VEL_WIN, pid))
                if p1 is None:
                    continue
                if p0 is not None:
                    vx = (p1[0] - p0[0]) / 12.0 / (VEL_WIN / 60.0)
                    vy = (p1[1] - p0[1]) / 12.0 / (VEL_WIN / 60.0)
                else:
                    vx = vy = 0.0
                is_def = pteam.get(pid) != hteam
                players.append((p1[0] / 12.0, p1[1] / 12.0, vx, vy, is_def))
                if is_def:
                    def_pos.append(p1)
                elif pid != hid:
                    off_other.append(p1)
            if len(players) < 6 or not def_pos:
                continue
            pcv = def_control((hpos[0] / 12.0, hpos[1] / 12.0), players)
            gaps = [min(math.hypot(o[0] - d[0], o[1] - d[1]) / 12.0 for d in def_pos) for o in off_other]
            n_open = sum(1 for x in gaps if x > OPEN_GAP_FT)
            min_kick = min(gaps) if gaps else float("nan")
            # --- defender pose + temporal at the catch (for Action-3D) ---
            did = int(r.nearest_def_id)
            bz = ball_z_map.get(cf)
            pose = pose_at(jmap.get((cf, did)), ball_z=bz)
            d15 = pos.get((cf - 15, did)); dnow = pos.get((cf, did))
            lat = (math.hypot(dnow[0] - d15[0], dnow[1] - d15[1]) / 12.0 / (15 / 60.0)
                   if d15 and dnow else "")
            mjd = _min_joint_dist_ft(jmap.get((cf, did)), jmap.get((cf, hid)))
            rows.append({
                "game_file": gf, "catch_frame": cf, "handler_id": hid,
                "nearest_def_id": did, "action": action,
                "def_dist_at_catch_ft": round(r.def_dist_ft, 3),
                "pitch_control_value": round(pcv, 4),
                "def_hand_elev_above_ball_in": pose.get("def_hand_elev_above_ball_in", ""),
                "arm_extension_ratio": pose.get("arm_extension_ratio", ""),
                "trunk_lean_deg": pose.get("trunk_lean_deg", ""),
                "lateral_velocity_ft_s": round(lat, 3) if lat != "" else "",
                "min_def_joint_dist_ft": mjd,
                "n_open_3pt_threats": n_open,
                "min_kickout_defender_gap_ft": round(min_kick, 3) if gaps else "",
                "release_shot_clock": round(sc.get(cf), 2) if sc.get(cf) is not None else "",
                "game_clock_seconds": round(gc.get(cf), 1) if gc.get(cf) is not None else "",
                "catch_to_release_s": round(cr, 3) if (isinstance(cr, float) and not math.isnan(cr)) else "",
            })
            n_ok += 1
        print(f"  {gf}: {n_ok} touch features")
        del pos, t

    cols = ["game_file", "catch_frame", "handler_id", "nearest_def_id", "action",
            "def_dist_at_catch_ft", "pitch_control_value", "def_hand_elev_above_ball_in",
            "arm_extension_ratio", "trunk_lean_deg", "lateral_velocity_ft_s",
            "min_def_joint_dist_ft", "n_open_3pt_threats",
            "min_kickout_defender_gap_ft", "release_shot_clock", "game_clock_seconds",
            "catch_to_release_s"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f"\nWrote {len(rows)} touches to {args.output_csv}")
    print("action distribution:", dict(Counter(r["action"] for r in rows)))


if __name__ == "__main__":
    main()
