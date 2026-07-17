#!/usr/bin/env python3
"""
Extract full 10-player kinematic state at each shot's release frame.

This is the input to the physics-based pitch-control model. Pitch control needs
every player's POSITION and VELOCITY (not just the nearest defender's, which is
all extract_shot_context_features.py saved). We make one pass over the raw
parquets and write a compact long table -- 10 rows per shot -- so the
pitch-control model can iterate without ever touching the ~1 GB parquets again.

For each shot we record, at the release frame, for all 10 players on court:
  position (feet, raw Hawk-Eye court frame), velocity (ft/s over a 0.25 s window),
  team membership, and shooter / nearest-defender flags.

We also record the nearest defender's PRE-SHOT position (release - 120 frames,
i.e. ~2 s earlier). The pitch-control counterfactual freezes the defender there
to measure how much space control the closeout itself added.

Output: data/intermediate/player_states_at_release.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)

VELOCITY_WINDOW_FRAMES = 15      # 0.25 s at 60 fps
PRE_SHOT_OFFSET_FRAMES = 120     # 2.0 s at 60 fps -- counterfactual freeze point
FPS = 60.0
FRAME_SEARCH_RADIUS = 5
IN_PER_FT = 12.0

PosMap = Dict[Tuple[int, int], Tuple[float, float]]
PlayerMap = Dict[int, Tuple[int, str]]


def _to_int(v: str) -> Optional[int]:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_shot_keys(shots_csv: str):
    by_game: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    with open(shots_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gf = (row.get("game_file") or "").strip()
            rf = _to_int(row.get("release_frame", ""))
            sid = _to_int(row.get("shooter_id", ""))
            did = _to_int(row.get("nearest_defender_id", ""))
            if gf and rf is not None and sid is not None and did is not None:
                by_game[gf].append((rf, sid, did))
    return by_game


def load_game_maps(path: str) -> Tuple[PosMap, PlayerMap, Optional[int]]:
    cols = ["frame", "player_id", "centroid_x", "centroid_y", "team_id", "teamName"]
    t = pq.read_table(path, columns=cols)
    frame = t["frame"].to_numpy(zero_copy_only=False)
    pid = t["player_id"].to_numpy(zero_copy_only=False)
    cx = t["centroid_x"].to_numpy(zero_copy_only=False)
    cy = t["centroid_y"].to_numpy(zero_copy_only=False)
    tid = t["team_id"].to_numpy(zero_copy_only=False)
    tnm = t["teamName"].to_pylist()
    pos_map: PosMap = {}
    player_map: PlayerMap = {}
    heat_team_id: Optional[int] = None
    for f, p, x, y, ti, tn in zip(frame, pid, cx, cy, tid, tnm):
        ip = int(p)
        pos_map[(int(f), ip)] = (float(x), float(y))
        if ip not in player_map:
            player_map[ip] = (int(ti), tn)
            if tn == "Heat":
                heat_team_id = int(ti)
    return pos_map, player_map, heat_team_id


def get_pos(pos_map: PosMap, frame: int, pid: int,
            radius: int = FRAME_SEARCH_RADIUS) -> Optional[Tuple[float, float]]:
    hit = pos_map.get((frame, pid))
    if hit is not None:
        return hit
    for d in range(1, radius + 1):
        hit = pos_map.get((frame - d, pid))
        if hit is not None:
            return hit
        hit = pos_map.get((frame + d, pid))
        if hit is not None:
            return hit
    return None


def velocity_ft_s(pos_map: PosMap, frame: int, pid: int) -> Tuple[float, float]:
    p1 = get_pos(pos_map, frame, pid)
    p0 = get_pos(pos_map, frame - VELOCITY_WINDOW_FRAMES, pid)
    if p0 is None or p1 is None:
        return (0.0, 0.0)
    dt = VELOCITY_WINDOW_FRAMES / FPS
    return ((p1[0] - p0[0]) / IN_PER_FT / dt, (p1[1] - p0[1]) / IN_PER_FT / dt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract 10-player kinematic state at release frame.")
    ap.add_argument("--shots-csv", default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default="data/intermediate/player_states_at_release.csv")
    args = ap.parse_args()

    by_game = load_shot_keys(args.shots_csv)
    total = sum(len(v) for v in by_game.values())
    print(f"Loaded {total} shots across {len(by_game)} games")

    rows: List[Dict[str, object]] = []
    n_bad = 0
    for gf in sorted(by_game):
        path = os.path.join(args.input_dir, gf)
        if not os.path.exists(path):
            print(f"  WARNING: missing parquet {gf}")
            continue
        pos_map, player_map, heat_id = load_game_maps(path)
        if heat_id is None:
            print(f"  WARNING: no Heat in {gf}")
            del pos_map, player_map
            continue

        for rf, sid, did in by_game[gf]:
            # Defender pre-shot (counterfactual freeze) position.
            def_pre = get_pos(pos_map, rf - PRE_SHOT_OFFSET_FRAMES, did)
            def_pre_x = def_pre[0] / IN_PER_FT if def_pre else ""
            def_pre_y = def_pre[1] / IN_PER_FT if def_pre else ""

            # All players on court at release (those with a tracked position).
            on_court = []
            for pid, (tid, _) in player_map.items():
                p = get_pos(pos_map, rf, pid)
                if p is None:
                    continue
                vx, vy = velocity_ft_s(pos_map, rf, pid)
                on_court.append((pid, tid, p[0] / IN_PER_FT, p[1] / IN_PER_FT, vx, vy))

            if len(on_court) != 10:
                n_bad += 1

            for pid, tid, x, y, vx, vy in on_court:
                rows.append({
                    "game_file": gf,
                    "release_frame": rf,
                    "shooter_id": sid,
                    "nearest_defender_id": did,
                    "player_id": pid,
                    "team_id": tid,
                    "is_offense": 0 if tid == heat_id else 1,
                    "is_shooter": 1 if pid == sid else 0,
                    "is_nearest_defender": 1 if pid == did else 0,
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "vx_ft_s": round(vx, 3),
                    "vy_ft_s": round(vy, 3),
                    "def_preshot_x_ft": round(def_pre_x, 3) if def_pre_x != "" else "",
                    "def_preshot_y_ft": round(def_pre_y, 3) if def_pre_y != "" else "",
                })
        print(f"  {gf}: {len(by_game[gf])} shots")
        del pos_map, player_map

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = ["game_file", "release_frame", "shooter_id", "nearest_defender_id",
                  "player_id", "team_id", "is_offense", "is_shooter", "is_nearest_defender",
                  "x_ft", "y_ft", "vx_ft_s", "vy_ft_s", "def_preshot_x_ft", "def_preshot_y_ft"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} player-rows to {args.output_csv} "
          f"({n_bad} shots did not have exactly 10 players on court)")


if __name__ == "__main__":
    main()
