#!/usr/bin/env python3
"""
Extract MIAMI HEAT 3-point attempts (Heat on offense, opponent defending).

This roughly DOUBLES the contested-3PA sample for the shot-value / pitch-control
validation. The existing dataset is opponent 3PA (Heat defending); here we flip
it: Heat shooters, opponent nearest-defender. Geometry/contest features are
computed identically to the opponent pipeline, and make/miss is labeled with the
tracking-based detector (validated at 97% vs play-by-play in
detect_make_from_tracking.py) since NBA play-by-play is unreachable here.

These shots have OPPONENT defenders, so they do NOT enter the per-Heat-defender
lift table. They are training/validation data for the team-agnostic
geometry->outcome and pitch-control->outcome relationships.

Output columns mirror the subset of shot_contest_dataset.csv needed downstream
(player-state extraction + pitch control + contest analysis), plus a
tracking_made label and pbp_shot_result set from tracking ("Made Shot"/"Missed Shot").

Output: data/intermediate/heat_3pa_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)

RIMS_XY = ((516.0, 0.0), (-516.0, 0.0))
RIM_Z = 120.0
THREEPT_MIN_INCHES = 264.0
# Tracking make-detector params (match detect_make_from_tracking.py).
SEARCH_AHEAD, SEARCH_BEHIND, NEAR_RIM_IN, MADE_THRESHOLD_IN = 170, 5, 24.0, 17.5


@dataclass
class HeatShot:
    game_file: str
    game_id: str
    opponent: str
    period: int
    release_frame: int
    release_shot_clock: str
    shooter_id: int
    shooter_name: str
    shooter_dist_to_rim_in: float
    rim_x: float
    rim_y: float
    release_ball_z: float
    nearest_defender_id: int
    nearest_defender_name: str
    contest_distance_ft: float
    closeout_speed_ft_s: float
    closeout_delta_ft_500ms: float
    contest_angle_deg: float
    hand_up_in: float
    shot_contest_quality: float
    rim_crossing_horiz_in: float
    tracking_made: int
    pbp_shot_result: str


def _norm(vx, vy):
    return math.sqrt(vx * vx + vy * vy)


def _angle_deg(ax, ay, bx, by):
    na, nb = _norm(ax, ay), _norm(bx, by)
    if na == 0 or nb == 0:
        return 180.0
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(c))


def _contest_quality(dist_ft, speed, angle, hand):
    d = max(0.0, 1.0 - dist_ft / 10.0)
    s = max(0.0, min(1.0, speed / 8.0))
    a = max(0.0, 1.0 - angle / 90.0)
    h = max(0.0, min(1.0, hand / 18.0))
    return round(100.0 * (0.35 * d + 0.30 * s + 0.20 * a + 0.15 * h), 2)


def _load_game_tables(path: str):
    cols = ["frame", "period", "gameClockTime", "shotClockTime", "ball_x", "ball_y",
            "ball_z", "last_touch_player_id", "player_id", "fullName", "team_id",
            "teamName", "centroid_x", "centroid_y", "centroid_z", "lWrist_z", "rWrist_z"]
    t = pq.read_table(path, columns=cols)
    players = t.select(["player_id", "fullName", "team_id", "teamName"]).group_by(
        ["player_id", "fullName", "team_id", "teamName"]).aggregate([])
    player_map: Dict[int, Tuple[str, int, str]] = {}
    for r in players.to_pylist():
        player_map[int(r["player_id"])] = (r["fullName"], int(r["team_id"]), r["teamName"])
    frame_tbl = t.select(["frame", "period", "gameClockTime", "shotClockTime", "ball_x",
                          "ball_y", "ball_z", "last_touch_player_id"]).group_by(
        ["frame", "period", "gameClockTime", "shotClockTime", "ball_x", "ball_y",
         "ball_z", "last_touch_player_id"]).aggregate([])
    idx = pc.sort_indices(frame_tbl, sort_keys=[("frame", "ascending")])
    frame_tbl = pc.take(frame_tbl, idx)
    pos_tbl = t.select(["frame", "player_id", "centroid_x", "centroid_y", "centroid_z",
                        "lWrist_z", "rWrist_z"]).group_by(
        ["frame", "player_id", "centroid_x", "centroid_y", "centroid_z",
         "lWrist_z", "rWrist_z"]).aggregate([])
    pos_map: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
    for r in pos_tbl.to_pylist():
        pos_map[(int(r["frame"]), int(r["player_id"]))] = (
            float(r["centroid_x"]), float(r["centroid_y"]), float(r["centroid_z"]),
            float(r["lWrist_z"]) if r["lWrist_z"] is not None else float("nan"),
            float(r["rWrist_z"]) if r["rWrist_z"] is not None else float("nan"))
    return frame_tbl, player_map, pos_map


def _detect_make(ball_x, ball_y, ball_z, frames, i, rim_x, rim_y):
    """Tracking-based make detection over the post-release window (by index)."""
    best = float("inf")
    end = min(len(frames) - 1, i + SEARCH_AHEAD)
    for j in range(max(0, i - SEARCH_BEHIND), end):
        if ball_z[j] >= RIM_Z >= ball_z[j + 1]:
            for jj in range(max(0, j - 2), min(len(frames), j + 3)):
                hd = math.hypot(ball_x[jj] - rim_x, ball_y[jj] - rim_y)
                if hd < best:
                    best = hd
    if not math.isfinite(best) or best > NEAR_RIM_IN:
        return 0, (best if math.isfinite(best) else 99.0)
    return (1 if best <= MADE_THRESHOLD_IN else 0), best


def _extract_one_game(path: str) -> List[HeatShot]:
    frame_tbl, player_map, pos_map = _load_game_tables(path)
    team_names = sorted({v[2] for v in player_map.values()})
    if "Heat" not in team_names:
        return []
    heat_team_id = next(v[1] for v in player_map.values() if v[2] == "Heat")
    opponent = next((nm for nm in team_names if nm != "Heat"), "Unknown")
    game_id = os.path.basename(path).replace("nba_game_", "").replace("_processed.parquet", "")

    frames = np.array(frame_tbl["frame"], dtype=np.int64)
    periods = np.array(frame_tbl["period"], dtype=np.int64)
    shot_clock = np.array(frame_tbl["shotClockTime"])
    ball_x = np.array(frame_tbl["ball_x"], dtype=float)
    ball_y = np.array(frame_tbl["ball_y"], dtype=float)
    ball_z = np.array(frame_tbl["ball_z"], dtype=float)
    last_touch = np.array(frame_tbl["last_touch_player_id"])

    events: List[HeatShot] = []
    last_idx = -999
    for i in range(3, len(frames) - 120):
        if i - last_idx < 20:
            continue
        if not (ball_z[i - 1] < 90.0 <= ball_z[i] and ball_z[i + 2] > ball_z[i]):
            continue
        shooter_raw = last_touch[i]
        if shooter_raw is None or (isinstance(shooter_raw, float) and np.isnan(shooter_raw)):
            continue
        shooter_id = int(shooter_raw)
        info = player_map.get(shooter_id)
        if info is None:
            continue
        shooter_name, shooter_team_id, _ = info
        if shooter_team_id != heat_team_id:   # FLIPPED: keep only Heat shooters
            continue

        window = slice(i, i + 120)
        rim_dmins = []
        for rx, ry in RIMS_XY:
            d = np.sqrt((ball_x[window] - rx) ** 2 + (ball_y[window] - ry) ** 2 + (ball_z[window] - RIM_Z) ** 2)
            rim_dmins.append(float(np.nanmin(d)))
        min_d = min(rim_dmins)
        if min_d > 18.0:
            continue
        rim_x, rim_y = RIMS_XY[0 if rim_dmins[0] < rim_dmins[1] else 1]

        shooter_pos = pos_map.get((int(frames[i]), shooter_id))
        if shooter_pos is None:
            continue
        sx, sy, _, _, _ = shooter_pos
        shooter_dist = math.hypot(sx - rim_x, sy - rim_y)
        if shooter_dist < THREEPT_MIN_INCHES:
            continue

        # FLIPPED: defenders are OPPONENT players.
        best_id, best_name, best_dist = None, "", float("inf")
        for pid, (name, tid, _) in player_map.items():
            if tid == heat_team_id:
                continue
            dpos = pos_map.get((int(frames[i]), pid))
            if dpos is None:
                continue
            dd = math.hypot(dpos[0] - sx, dpos[1] - sy)
            if dd < best_dist:
                best_dist, best_id, best_name = dd, pid, name
        if best_id is None:
            continue

        prior_idx = max(0, i - 30)
        prior_def = pos_map.get((int(frames[prior_idx]), best_id))
        closeout_delta_ft = closeout_speed = 0.0
        if prior_def is not None:
            prior_dist_in = math.hypot(prior_def[0] - sx, prior_def[1] - sy)
            closeout_delta_ft = (prior_dist_in - best_dist) / 12.0
            closeout_speed = closeout_delta_ft / 0.5

        def_now = pos_map[(int(frames[i]), best_id)]
        angle = _angle_deg(sx - def_now[0], sy - def_now[1], rim_x - sx, rim_y - sy)
        hand_up = 0.0
        w = [def_now[3], def_now[4]]
        valid = [x for x in w if not math.isnan(x)]
        if valid:
            hand_up = max(valid) - def_now[2]

        made, horiz = _detect_make(ball_x, ball_y, ball_z, frames, i, rim_x, rim_y)
        contest_distance_ft = best_dist / 12.0
        events.append(HeatShot(
            game_file=os.path.basename(path), game_id=game_id, opponent=opponent,
            period=int(periods[i]), release_frame=int(frames[i]),
            release_shot_clock=str(shot_clock[i]), shooter_id=shooter_id,
            shooter_name=shooter_name, shooter_dist_to_rim_in=round(shooter_dist, 2),
            rim_x=rim_x, rim_y=rim_y, release_ball_z=round(float(ball_z[i]), 2),
            nearest_defender_id=best_id, nearest_defender_name=best_name,
            contest_distance_ft=round(contest_distance_ft, 3),
            closeout_speed_ft_s=round(closeout_speed, 3),
            closeout_delta_ft_500ms=round(closeout_delta_ft, 3),
            contest_angle_deg=round(angle, 2), hand_up_in=round(hand_up, 2),
            shot_contest_quality=_contest_quality(contest_distance_ft, closeout_speed, angle, hand_up),
            rim_crossing_horiz_in=round(horiz, 2), tracking_made=made,
            pbp_shot_result="Made Shot" if made else "Missed Shot"))
        last_idx = i
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default="data/intermediate/heat_3pa_dataset.csv")
    args = ap.parse_args()

    files = sorted(os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if f.endswith(".parquet"))
    all_events: List[HeatShot] = []
    for fp in files:
        ev = _extract_one_game(fp)
        all_events.extend(ev)
        n_made = sum(e.tracking_made for e in ev)
        print(f"  {os.path.basename(fp)}: {len(ev)} Heat 3PA ({n_made} made)")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in asdict(all_events[0]).keys()])
        w.writeheader()
        for e in all_events:
            w.writerow(asdict(e))
    made = sum(e.tracking_made for e in all_events)
    print(f"\nWrote {len(all_events)} Heat 3PA to {args.output_csv} "
          f"(tracking make rate {made/len(all_events):.3f})")


if __name__ == "__main__":
    main()
