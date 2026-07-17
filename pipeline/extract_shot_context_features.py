#!/usr/bin/env python3
"""
extract_shot_context_features.py  --  Spec section 4.1

Extend each shot in shot_contest_dataset.csv with pre-shot positioning and
spatial context pulled from the raw Hawk-Eye parquets. Processes ONE parquet at
a time (each ~1.1 GB): groups shots by game_file, opens each parquet once,
extracts all shots for that game, then discards the maps before the next game.

Features per shot (spec 4.1):
  pre_shot_distance_ft         defender-to-shooter distance 2.0 s (120 frames)
                               before release -- positioning quality.
  reactive_speed_index         closeout_speed_ft_s / max(pre_shot_distance_ft, 0.5)
                               -- efficiency covering the gap (denominator capped
                               at 0.5 ft to avoid blow-up for stationed defenders).
  min_kickout_defender_gap_ft  most-open non-shooting offensive player's distance
                               to the nearest Heat defender at release.
  n_open_3pt_threats           count of those players > 5 ft from any defender.
  avg_off_defender_gap_ft      mean of those gaps.
  defender_lateral_speed_ft_s  nearest defender's planar speed over the 0.25 s
                               (15 frames) ending at release.
  pre_shot_data_missing        True if no tracking at pre_shot_frame (period start).
  stationed                    True if pre_shot_distance_ft < 0.5 ft.

Edge cases (spec 4.1):
  - pre_shot_frame before period data: pre_shot_data_missing=True, keep row,
    exclude from decomposition (pre_shot_distance/reactive_speed_index = NaN).
  - player missing at exact frame: use nearest available frame within +/-5.

Output: data/intermediate/shot_context_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
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

PRE_SHOT_OFFSET_FRAMES = 120   # 2.0 s at 60 fps
VELOCITY_WINDOW = 15           # 0.25 s at 60 fps
FPS = 60.0
OPEN_THREAT_GAP_FT = 5.0
FRAME_SEARCH_RADIUS = 5
REACTIVE_DENOM_CAP_FT = 0.5    # cap denominator; below this defender is "stationed"

PosMap = Dict[Tuple[int, int], Tuple[float, float]]
PlayerMap = Dict[int, Tuple[int, str]]   # player_id -> (team_id, teamName)


@dataclass
class ShotKey:
    game_file: str
    release_frame: int
    shooter_id: int
    nearest_defender_id: int
    closeout_speed_ft_s: float


@dataclass
class Features:
    game_file: str
    release_frame: int
    shooter_id: int
    nearest_defender_id: int
    pre_shot_distance_ft: float
    reactive_speed_index: float
    min_kickout_defender_gap_ft: float
    n_open_3pt_threats: int
    avg_off_defender_gap_ft: float
    defender_lateral_speed_ft_s: float
    pre_shot_data_missing: bool
    stationed: bool


def _to_int(v: str) -> Optional[int]:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_float(v: str) -> float:
    s = (v or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_shot_keys(shots_csv: str) -> Dict[str, List[ShotKey]]:
    by_game: Dict[str, List[ShotKey]] = defaultdict(list)
    with open(shots_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"game_file", "release_frame", "shooter_id", "nearest_defender_id", "closeout_speed_ft_s"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"shots CSV missing columns: {sorted(missing)}")
        for row in reader:
            gf = (row.get("game_file") or "").strip()
            rf = _to_int(row.get("release_frame"))
            sid = _to_int(row.get("shooter_id"))
            did = _to_int(row.get("nearest_defender_id"))
            if not gf or rf is None or sid is None or did is None:
                continue
            by_game[gf].append(ShotKey(gf, rf, sid, did, _to_float(row.get("closeout_speed_ft_s"))))
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


def get_pos(pos_map: PosMap, frame: int, pid: int, radius: int = FRAME_SEARCH_RADIUS):
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


def compute(key: ShotKey, pos_map: PosMap, player_map: PlayerMap, heat_team_id: int) -> Features:
    rf, sid, did = key.release_frame, key.shooter_id, key.nearest_defender_id

    # 1. Pre-shot positioning + reactive speed
    pre_frame = rf - PRE_SHOT_OFFSET_FRAMES
    dpre = get_pos(pos_map, pre_frame, did)
    spre = get_pos(pos_map, pre_frame, sid)
    if dpre is None or spre is None:
        pre_dist = float("nan"); reactive = float("nan"); missing = True; stationed = False
    else:
        pre_dist = math.hypot(dpre[0] - spre[0], dpre[1] - spre[1]) / 12.0
        missing = False
        stationed = pre_dist < REACTIVE_DENOM_CAP_FT
        denom = max(pre_dist, REACTIVE_DENOM_CAP_FT)
        reactive = key.closeout_speed_ft_s / denom if math.isfinite(key.closeout_speed_ft_s) else float("nan")

    # 2. Spatial context at release: non-shooting offense -> nearest Heat defender
    offense = [p for p, (ti, _) in player_map.items() if ti != heat_team_id and p != sid]
    defenders = [p for p, (ti, _) in player_map.items() if ti == heat_team_id]
    def_pos = [dp for dp in (get_pos(pos_map, rf, d) for d in defenders) if dp is not None]
    gaps: List[float] = []
    for op in offense:
        o = get_pos(pos_map, rf, op)
        if o is None or not def_pos:
            continue
        gaps.append(min(math.hypot(o[0] - d[0], o[1] - d[1]) / 12.0 for d in def_pos))
    if gaps:
        min_gap = min(gaps); n_open = sum(1 for g in gaps if g > OPEN_THREAT_GAP_FT); avg_gap = sum(gaps) / len(gaps)
    else:
        min_gap = float("nan"); n_open = 0; avg_gap = float("nan")

    # 3. Defender lateral (planar) speed at release
    p0 = get_pos(pos_map, rf - VELOCITY_WINDOW, did)
    p1 = get_pos(pos_map, rf, did)
    if p0 is not None and p1 is not None:
        dt = VELOCITY_WINDOW / FPS
        lat = math.hypot((p1[0] - p0[0]) / 12.0, (p1[1] - p0[1]) / 12.0) / dt
    else:
        lat = float("nan")

    return Features(key.game_file, rf, sid, did, pre_dist, reactive, min_gap, n_open,
                    avg_gap, lat, missing, stationed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract pre-shot positioning + spatial context (spec 4.1).")
    ap.add_argument("--shots-csv", default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-csv", default="data/intermediate/shot_context_features.csv")
    args = ap.parse_args()

    by_game = load_shot_keys(args.shots_csv)
    print(f"Loaded {sum(len(v) for v in by_game.values())} shots across {len(by_game)} games")
    rows: List[Features] = []
    for gf in sorted(by_game):
        path = os.path.join(args.input_dir, gf)
        if not os.path.exists(path):
            print(f"  WARNING: missing parquet {gf} ({len(by_game[gf])} shots skipped)"); continue
        pos_map, player_map, heat_team_id = load_game_maps(path)
        if heat_team_id is None:
            print(f"  WARNING: no Heat team in {gf}"); del pos_map, player_map; continue
        n_missing = 0
        for key in by_game[gf]:
            feats = compute(key, pos_map, player_map, heat_team_id)
            n_missing += feats.pre_shot_data_missing
            rows.append(feats)
        print(f"  {gf}: {len(by_game[gf])} shots ({n_missing} pre-shot missing)")
        del pos_map, player_map

    fieldnames = ["game_file", "release_frame", "shooter_id", "nearest_defender_id",
                  "pre_shot_distance_ft", "reactive_speed_index", "min_kickout_defender_gap_ft",
                  "n_open_3pt_threats", "avg_off_defender_gap_ft", "defender_lateral_speed_ft_s",
                  "pre_shot_data_missing", "stationed"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "game_file": r.game_file, "release_frame": r.release_frame,
                "shooter_id": r.shooter_id, "nearest_defender_id": r.nearest_defender_id,
                "pre_shot_distance_ft": _fmt(r.pre_shot_distance_ft),
                "reactive_speed_index": _fmt(r.reactive_speed_index),
                "min_kickout_defender_gap_ft": _fmt(r.min_kickout_defender_gap_ft),
                "n_open_3pt_threats": r.n_open_3pt_threats,
                "avg_off_defender_gap_ft": _fmt(r.avg_off_defender_gap_ft),
                "defender_lateral_speed_ft_s": _fmt(r.defender_lateral_speed_ft_s),
                "pre_shot_data_missing": "yes" if r.pre_shot_data_missing else "no",
                "stationed": "yes" if r.stationed else "no",
            })
    n_miss = sum(r.pre_shot_data_missing for r in rows)
    n_stat = sum(r.stationed for r in rows)
    print(f"\nWrote {len(rows)} rows to {args.output_csv} "
          f"({n_miss} pre_shot_data_missing, {n_stat} stationed)")


if __name__ == "__main__":
    main()
