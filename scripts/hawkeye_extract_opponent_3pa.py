#!/usr/bin/env python3
"""
Extract opponent three-point attempts from NBA Hawk-Eye tracking parquet files.

The script processes one game file at a time (memory-safe for ~1GB parquet files),
identifies likely shot trajectories, filters to opponent 3PA, and writes per-shot
features for contest analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Allow dependency installs in workspace-local .python_packages.
LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np


RIMS_XY = ((516.0, 0.0), (-516.0, 0.0))  # inches, Hawk-Eye court frame
RIM_Z = 120.0  # 10 ft in inches
THREEPT_MIN_INCHES = 264.0  # 22 ft; tune later for arc/corner exactness


@dataclass
class ShotEvent:
    game_file: str
    opponent: str
    frame: int
    period: int
    game_clock: str
    shot_clock: str
    shooter_id: int
    shooter_name: str
    shooter_team: str
    shooter_dist_to_rim_in: float
    rim_x: float
    rim_y: float
    release_ball_x: float
    release_ball_y: float
    release_ball_z: float
    min_ball_rim_3d_in: float
    nearest_defender_id: int
    nearest_defender_name: str
    contest_distance_ft: float
    closeout_speed_ft_s: float
    closeout_delta_ft_500ms: float
    contest_angle_deg: float
    hand_up_in: float
    shot_contest_quality: float


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def _norm(vx: float, vy: float) -> float:
    return math.sqrt(vx * vx + vy * vy)


def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    na = _norm(ax, ay)
    nb = _norm(bx, by)
    if na == 0.0 or nb == 0.0:
        return 180.0
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(c))


def _contest_quality(
    contest_distance_ft: float,
    closeout_speed_ft_s: float,
    contest_angle_deg: float,
    hand_up_in: float,
) -> float:
    """Heuristic 0-100 score; intended as a tunable starting point."""
    distance_component = max(0.0, 1.0 - contest_distance_ft / 10.0)  # best <= 3-4 ft
    speed_component = max(0.0, min(1.0, closeout_speed_ft_s / 8.0))  # strong closeout ~8+ ft/s
    angle_component = max(0.0, 1.0 - contest_angle_deg / 90.0)  # inline (0 deg) better
    hand_component = max(0.0, min(1.0, hand_up_in / 18.0))  # hand raised ~18in above torso
    score = (
        0.35 * distance_component
        + 0.30 * speed_component
        + 0.20 * angle_component
        + 0.15 * hand_component
    )
    return round(100.0 * score, 2)


def _load_game_tables(path: str):
    cols = [
        "frame",
        "period",
        "gameClockTime",
        "shotClockTime",
        "ball_x",
        "ball_y",
        "ball_z",
        "last_touch_player_id",
        "player_id",
        "fullName",
        "team_id",
        "teamName",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "lWrist_z",
        "rWrist_z",
    ]
    t = pq.read_table(path, columns=cols)

    players = t.select(["player_id", "fullName", "team_id", "teamName"]).group_by(
        ["player_id", "fullName", "team_id", "teamName"]
    ).aggregate([])
    player_map: Dict[int, Tuple[str, int, str]] = {}
    for r in players.to_pylist():
        player_map[int(r["player_id"])] = (r["fullName"], int(r["team_id"]), r["teamName"])

    frame_tbl = t.select(
        [
            "frame",
            "period",
            "gameClockTime",
            "shotClockTime",
            "ball_x",
            "ball_y",
            "ball_z",
            "last_touch_player_id",
        ]
    ).group_by(
        [
            "frame",
            "period",
            "gameClockTime",
            "shotClockTime",
            "ball_x",
            "ball_y",
            "ball_z",
            "last_touch_player_id",
        ]
    ).aggregate([])
    idx = pc.sort_indices(frame_tbl, sort_keys=[("frame", "ascending")])
    frame_tbl = pc.take(frame_tbl, idx)

    # Per-frame player position snapshot.
    pos_tbl = t.select(
        ["frame", "player_id", "centroid_x", "centroid_y", "centroid_z", "lWrist_z", "rWrist_z"]
    ).group_by(
        ["frame", "player_id", "centroid_x", "centroid_y", "centroid_z", "lWrist_z", "rWrist_z"]
    ).aggregate([])
    pos_map: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
    for r in pos_tbl.to_pylist():
        pos_map[(int(r["frame"]), int(r["player_id"]))] = (
            float(r["centroid_x"]),
            float(r["centroid_y"]),
            float(r["centroid_z"]),
            float(r["lWrist_z"]) if r["lWrist_z"] is not None else float("nan"),
            float(r["rWrist_z"]) if r["rWrist_z"] is not None else float("nan"),
        )
    return frame_tbl, player_map, pos_map


def _extract_one_game(path: str) -> List[ShotEvent]:
    frame_tbl, player_map, pos_map = _load_game_tables(path)
    team_names = sorted({v[2] for v in player_map.values()})
    if "Heat" not in team_names:
        return []
    heat_team_id = next(v[1] for v in player_map.values() if v[2] == "Heat")
    opponent = next((nm for nm in team_names if nm != "Heat"), "Unknown")

    frames = np.array(frame_tbl["frame"], dtype=np.int64)
    periods = np.array(frame_tbl["period"], dtype=np.int64)
    game_clock = np.array(frame_tbl["gameClockTime"])
    shot_clock = np.array(frame_tbl["shotClockTime"])
    ball_x = np.array(frame_tbl["ball_x"], dtype=float)
    ball_y = np.array(frame_tbl["ball_y"], dtype=float)
    ball_z = np.array(frame_tbl["ball_z"], dtype=float)
    last_touch = np.array(frame_tbl["last_touch_player_id"])

    events: List[ShotEvent] = []
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
        shooter_info = player_map.get(shooter_id)
        if shooter_info is None:
            continue
        shooter_name, shooter_team_id, shooter_team_name = shooter_info
        if shooter_team_id == heat_team_id:
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
        shooter_x, shooter_y, _, _, _ = shooter_pos
        shooter_dist = math.hypot(shooter_x - rim_x, shooter_y - rim_y)
        if shooter_dist < THREEPT_MIN_INCHES:
            continue

        # Heat is defending, so only Heat players are defender candidates.
        defender_candidates = [
            (pid, info)
            for pid, info in player_map.items()
            if info[1] == heat_team_id
        ]
        best_defender_id: Optional[int] = None
        best_defender_name = ""
        best_dist = float("inf")

        for pid, (name, _, _) in defender_candidates:
            dpos = pos_map.get((int(frames[i]), pid))
            if dpos is None:
                continue
            dx = dpos[0] - shooter_x
            dy = dpos[1] - shooter_y
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist = d
                best_defender_id = pid
                best_defender_name = name
        if best_defender_id is None:
            continue

        # Closeout speed over ~500 ms (30 frames at ~60Hz), if available.
        prior_idx = max(0, i - 30)
        prior_def = pos_map.get((int(frames[prior_idx]), best_defender_id))
        closeout_delta_ft = 0.0
        closeout_speed_ft_s = 0.0
        if prior_def is not None:
            prior_dist_in = math.hypot(prior_def[0] - shooter_x, prior_def[1] - shooter_y)
            closeout_delta_ft = (prior_dist_in - best_dist) / 12.0
            closeout_speed_ft_s = closeout_delta_ft / 0.5

        # Angle: defender->shooter vs shooter->rim.
        def_now = pos_map[(int(frames[i]), best_defender_id)]
        v1x, v1y = shooter_x - def_now[0], shooter_y - def_now[1]
        v2x, v2y = rim_x - shooter_x, rim_y - shooter_y
        angle = _angle_deg(v1x, v1y, v2x, v2y)

        # Hand-up proxy: max wrist height above torso.
        hand_up_in = 0.0
        if not math.isnan(def_now[3]) and not math.isnan(def_now[4]):
            hand_up_in = max(def_now[3], def_now[4]) - def_now[2]
        elif not math.isnan(def_now[3]):
            hand_up_in = def_now[3] - def_now[2]
        elif not math.isnan(def_now[4]):
            hand_up_in = def_now[4] - def_now[2]

        contest_distance_ft = best_dist / 12.0
        score = _contest_quality(contest_distance_ft, closeout_speed_ft_s, angle, hand_up_in)

        events.append(
            ShotEvent(
                game_file=os.path.basename(path),
                opponent=opponent,
                frame=int(frames[i]),
                period=int(periods[i]),
                game_clock=_safe_str(game_clock[i]),
                shot_clock=_safe_str(shot_clock[i]),
                shooter_id=shooter_id,
                shooter_name=shooter_name,
                shooter_team=shooter_team_name,
                shooter_dist_to_rim_in=round(shooter_dist, 2),
                rim_x=rim_x,
                rim_y=rim_y,
                release_ball_x=round(float(ball_x[i]), 3),
                release_ball_y=round(float(ball_y[i]), 3),
                release_ball_z=round(float(ball_z[i]), 3),
                min_ball_rim_3d_in=round(min_d, 2),
                nearest_defender_id=best_defender_id,
                nearest_defender_name=best_defender_name,
                contest_distance_ft=round(contest_distance_ft, 3),
                closeout_speed_ft_s=round(closeout_speed_ft_s, 3),
                closeout_delta_ft_500ms=round(closeout_delta_ft, 3),
                contest_angle_deg=round(angle, 2),
                hand_up_in=round(hand_up_in, 2),
                shot_contest_quality=score,
            )
        )
        last_idx = i

    return events


def _write_events(path: str, events: Iterable[ShotEvent]) -> None:
    fieldnames = [f.name for f in ShotEvent.__dataclass_fields__.values()]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            writer.writerow(e.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract opponent 3PA and contest features from Hawk-Eye parquet files.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing *_processed.parquet game files.",
    )
    parser.add_argument(
        "--output-csv",
        default="opponent_3pa_events_with_contest_features.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".parquet")
    )
    all_events: List[ShotEvent] = []
    for fp in files:
        game_events = _extract_one_game(fp)
        all_events.extend(game_events)
        print(f"{os.path.basename(fp)}: extracted {len(game_events)} opponent 3PA candidates")

    _write_events(args.output_csv, all_events)
    print(f"Wrote {len(all_events)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
