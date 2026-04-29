#!/usr/bin/env python3
"""
Build unified shot contest dataset from Hawk-Eye tracking + NBA play-by-play.

One row per opponent 3-point attempt. Columns:

  Identifiers
    game_file, game_id, opponent, period

  Release  (Hawk-Eye: moment ball leaves shooter's hand)
    release_frame, release_game_clock, release_shot_clock
    shooter_id, shooter_name, shooter_team
    shooter_dist_to_rim_in, rim_x, rim_y
    release_ball_x, release_ball_y, release_ball_z

  Arc apex  (Hawk-Eye: frame of maximum ball height after release)
    apex_frame, apex_game_clock, apex_shot_clock, apex_ball_z

  Contest features  (Hawk-Eye: computed at release frame)
    min_ball_rim_3d_in
    nearest_defender_id, nearest_defender_name
    contest_distance_ft, closeout_speed_ft_s, closeout_delta_ft_500ms
    contest_angle_deg, hand_up_in, shot_contest_quality

  PBP outcome  (NBA CDN: matched within 5 s of release, same player+period)
    pbp_shot_result, pbp_conclusion_game_clock, pbp_description
    (all three are "NA" when no PBP event matches)

Usage:
    python build_unified_shot_dataset.py \\
        --input-dir /path/to/onedrive/miami_heat_2025 \\
        --output-csv shot_contest_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIMS_XY = ((516.0, 0.0), (-516.0, 0.0))   # inches, Hawk-Eye court frame
RIM_Z = 120.0                               # 10 ft in inches
THREEPT_MIN_INCHES = 264.0                  # ~22 ft; minimum distance for a 3PA
RELEASE_Z_THRESHOLD = 90.0                  # ball must cross this height (inches) while rising
APEX_SEARCH_FRAMES = 120                    # scan up to ~2 s after release for arc apex
PBP_MATCH_WINDOW_SEC = 5.0                  # search up to 5 s after release in PBP
PBP_CLOCK_BUFFER_SEC = 1.0                  # allow 1 s before release (clock sync tolerance)

MIAMI_TEAM_ID = 1610612748
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
NA = "NA"


# ---------------------------------------------------------------------------
# Output columns (order matters for CSV)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    # Identifiers
    "game_file", "game_id", "opponent", "period",
    # Release
    "release_frame", "release_game_clock", "release_shot_clock",
    "shooter_id", "shooter_name", "shooter_team",
    "shooter_dist_to_rim_in", "rim_x", "rim_y",
    "release_ball_x", "release_ball_y", "release_ball_z",
    # Arc apex
    "apex_frame", "apex_game_clock", "apex_shot_clock", "apex_ball_z",
    # Contest features
    "min_ball_rim_3d_in",
    "nearest_defender_id", "nearest_defender_name",
    "contest_distance_ft", "closeout_speed_ft_s", "closeout_delta_ft_500ms",
    "contest_angle_deg", "hand_up_in", "shot_contest_quality",
    # PBP outcome
    "pbp_shot_result", "pbp_conclusion_game_clock", "pbp_description",
]


# ---------------------------------------------------------------------------
# Utility: game ID from filename
# ---------------------------------------------------------------------------

def _game_id_from_filename(name: str) -> Optional[str]:
    """nba_game_0022500062_processed.parquet -> '0022500062'"""
    m = re.search(r"(\d{10})", name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Utility: clock conversions
# ---------------------------------------------------------------------------

def _parquet_clock_to_sec(clock_val) -> float:
    """
    Parquet stores game clock as MM:SS (e.g. '11:41') for most of the quarter,
    and as a plain float string (e.g. '45.0') for the final minute.
    Returns remaining seconds as a float, or nan if unparseable.
    """
    if clock_val is None:
        return float("nan")
    s = str(clock_val).strip()
    if not s or s in ("None", "nan"):
        return float("nan")
    if ":" in s:
        mins, secs = s.split(":", 1)
        return int(mins) * 60.0 + float(secs)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _iso_clock_to_sec(iso: str) -> float:
    """PT10M31.00S -> 631.0"""
    m = re.fullmatch(r"PT(?P<m>\d+)M(?P<s>\d+\.?\d*)S", (iso or "").strip())
    if not m:
        return float("nan")
    return int(m["m"]) * 60.0 + float(m["s"])


def _sec_to_mmss(sec: float) -> str:
    """631.0 -> '10:31'  |  45.5 -> '0:45'"""
    if math.isnan(sec) or sec < 0:
        return NA
    mins = int(sec) // 60
    secs = int(sec % 60)
    return f"{mins}:{secs:02d}"


def _safe_clock_str(val) -> str:
    """Return the raw parquet clock string, or NA if missing."""
    if val is None:
        return NA
    s = str(val).strip()
    return s if s and s not in ("None", "nan") else NA


# ---------------------------------------------------------------------------
# PBP: fetch + parse
# ---------------------------------------------------------------------------

def _fetch_pbp(game_id: str) -> List[dict]:
    url = PBP_URL.format(game_id=game_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["game"]["actions"]
    except Exception as exc:
        print(f"  WARNING: PBP fetch failed for {game_id}: {exc}")
        return []


def _parse_pbp_opponent_3pa(actions: List[dict]) -> List[dict]:
    """
    Filter NBA CDN play-by-play to opponent 3-point attempts only.
    Returns one dict per event with the fields we need for matching.
    """
    out: List[dict] = []
    for a in actions:
        if not a.get("isFieldGoal"):
            continue
        desc = a.get("description") or ""
        if "3PT" not in desc and "3-PT" not in desc:
            continue
        team_id = a.get("teamId")
        if team_id is None or int(team_id) == MIAMI_TEAM_ID:
            continue
        pid = a.get("personId")
        if not pid:
            continue
        remaining = _iso_clock_to_sec(a.get("clock") or "")
        out.append({
            "period":        int(a["period"]),
            "person_id":     int(pid),
            "player_abbr":   (a.get("playerNameI") or "").strip(),
            "remaining_sec": remaining,
            "shot_result":   (a.get("shotResult") or "").strip(),
            "description":   desc.strip(),
        })
    return out


def _match_pbp(
    pbp_events: List[dict],
    shooter_id: int,
    period: int,
    release_sec: float,
) -> Tuple[Optional[dict], int]:
    """
    Find the PBP event for this shot.

    Game clock counts DOWN: release at 660 s, shot resolves ~3 s later at 657 s.
    We look in [release_sec - PBP_MATCH_WINDOW_SEC, release_sec + PBP_CLOCK_BUFFER_SEC]
    and take the closest match in time.

    Returns (matched_event, index_in_pbp_events) or (None, -1) if no match.
    """
    if math.isnan(release_sec):
        return None, -1
    lo = release_sec - PBP_MATCH_WINDOW_SEC
    hi = release_sec + PBP_CLOCK_BUFFER_SEC
    candidates = [
        (idx, e) for idx, e in enumerate(pbp_events)
        if e["period"] == period
        and e["person_id"] == shooter_id
        and lo <= e["remaining_sec"] <= hi
    ]
    if not candidates:
        return None, -1
    best_idx, best_event = min(candidates, key=lambda t: abs(t[1]["remaining_sec"] - release_sec))
    return best_event, best_idx


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def _load_game_tables(
    path: str,
) -> Tuple[object, Dict[int, Tuple[str, int, str]], Dict[Tuple[int, int], Tuple]]:
    cols = [
        "frame", "period", "gameClockTime", "shotClockTime",
        "ball_x", "ball_y", "ball_z", "last_touch_player_id",
        "player_id", "fullName", "team_id", "teamName",
        "centroid_x", "centroid_y", "centroid_z",
        "lWrist_z", "rWrist_z",
    ]
    t = pq.read_table(path, columns=cols)

    # Player directory: player_id -> (name, team_id, team_name)
    players = (
        t.select(["player_id", "fullName", "team_id", "teamName"])
        .group_by(["player_id", "fullName", "team_id", "teamName"])
        .aggregate([])
    )
    player_map: Dict[int, Tuple[str, int, str]] = {
        int(r["player_id"]): (r["fullName"], int(r["team_id"]), r["teamName"])
        for r in players.to_pylist()
    }

    # Per-frame ball snapshot (deduplicated)
    frame_cols = [
        "frame", "period", "gameClockTime", "shotClockTime",
        "ball_x", "ball_y", "ball_z", "last_touch_player_id",
    ]
    frame_tbl = (
        t.select(frame_cols)
        .group_by(frame_cols)
        .aggregate([])
    )
    idx = pc.sort_indices(frame_tbl, sort_keys=[("frame", "ascending")])
    frame_tbl = pc.take(frame_tbl, idx)

    # Per-frame player positions: (frame, player_id) -> (cx, cy, cz, lWrist_z, rWrist_z)
    pos_cols = ["frame", "player_id", "centroid_x", "centroid_y", "centroid_z", "lWrist_z", "rWrist_z"]
    pos_tbl = t.select(pos_cols).group_by(pos_cols).aggregate([])
    pos_map: Dict[Tuple[int, int], Tuple] = {}
    for r in pos_tbl.to_pylist():
        pos_map[(int(r["frame"]), int(r["player_id"]))] = (
            float(r["centroid_x"]),
            float(r["centroid_y"]),
            float(r["centroid_z"]),
            float(r["lWrist_z"]) if r["lWrist_z"] is not None else float("nan"),
            float(r["rWrist_z"]) if r["rWrist_z"] is not None else float("nan"),
        )

    return frame_tbl, player_map, pos_map


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """Angle between 2-D vectors (ax, ay) and (bx, by) in degrees."""
    na = math.sqrt(ax*ax + ay*ay)
    nb = math.sqrt(bx*bx + by*by)
    if na == 0.0 or nb == 0.0:
        return 180.0
    c = max(-1.0, min(1.0, (ax*bx + ay*by) / (na * nb)))
    return math.degrees(math.acos(c))


def _contest_quality(dist_ft: float, speed_ft_s: float, angle_deg: float, hand_in: float) -> float:
    """Heuristic 0-100 SCQ score. Weights are a starting point, not validated."""
    return round(100.0 * (
        0.35 * max(0.0, 1.0 - dist_ft / 10.0)
      + 0.30 * max(0.0, min(1.0, speed_ft_s / 8.0))
      + 0.20 * max(0.0, 1.0 - angle_deg / 90.0)
      + 0.15 * max(0.0, min(1.0, hand_in / 18.0))
    ), 2)


# ---------------------------------------------------------------------------
# Per-game extraction
# ---------------------------------------------------------------------------

def _extract_one_game(
    path: str, pbp_events: List[dict]
) -> Tuple[List[dict], set, str]:
    """
    Returns:
        rows             — one dict per tracking-detected shot
        matched_pbp_idx  — set of pbp_events indices that were matched to a tracking shot
        opponent         — opposing team name
    """
    frame_tbl, player_map, pos_map = _load_game_tables(path)

    team_names = sorted({v[2] for v in player_map.values()})
    if "Heat" not in team_names:
        return [], set(), "Unknown"
    heat_team_id = next(v[1] for v in player_map.values() if v[2] == "Heat")
    opponent = next((nm for nm in team_names if nm != "Heat"), "Unknown")

    # Build numpy arrays for fast frame-by-frame scanning
    frames     = np.array(frame_tbl["frame"],              dtype=np.int64)
    periods    = np.array(frame_tbl["period"],             dtype=np.int64)
    game_clks  = np.array(frame_tbl["gameClockTime"])      # object array; may be str or None
    shot_clks  = np.array(frame_tbl["shotClockTime"])      # object array
    ball_x     = np.array(frame_tbl["ball_x"],             dtype=float)
    ball_y     = np.array(frame_tbl["ball_y"],             dtype=float)
    ball_z     = np.array(frame_tbl["ball_z"],             dtype=float)
    last_touch = np.array(frame_tbl["last_touch_player_id"])

    rows: List[dict] = []
    matched_pbp_idx: set = set()
    last_shot_idx = -999

    for i in range(3, len(frames) - APEX_SEARCH_FRAMES):
        # Cooldown: ignore detections within 20 frames of the previous shot
        if i - last_shot_idx < 20:
            continue

        # ── Release detection ────────────────────────────────────────────────
        # Ball crosses RELEASE_Z_THRESHOLD (90 in / 7.5 ft) going upward,
        # and is still rising 2 frames later (rules out momentary spikes).
        if not (ball_z[i-1] < RELEASE_Z_THRESHOLD <= ball_z[i] and ball_z[i+2] > ball_z[i]):
            continue

        # ── Identify shooter ─────────────────────────────────────────────────
        raw_touch = last_touch[i]
        if raw_touch is None or (isinstance(raw_touch, float) and np.isnan(raw_touch)):
            continue
        shooter_id = int(raw_touch)
        shooter_info = player_map.get(shooter_id)
        if shooter_info is None:
            continue
        shooter_name, shooter_team_id, shooter_team_name = shooter_info
        if shooter_team_id == heat_team_id:
            continue   # Heat offensive shot — skip

        # ── Verify ball approaches a rim (not a dribble / pass) ──────────────
        # Scan the next APEX_SEARCH_FRAMES frames and check min 3-D distance to either rim.
        w = slice(i, i + APEX_SEARCH_FRAMES)
        rim_dists = []
        for rx, ry in RIMS_XY:
            d3 = np.sqrt(
                (ball_x[w] - rx)**2 + (ball_y[w] - ry)**2 + (ball_z[w] - RIM_Z)**2
            )
            rim_dists.append(float(np.nanmin(d3)))
        min_rim_dist = min(rim_dists)
        if min_rim_dist > 18.0:
            continue   # never got close to either rim — not a shot
        rim_x, rim_y = RIMS_XY[0 if rim_dists[0] < rim_dists[1] else 1]

        # ── Verify 3-point distance ──────────────────────────────────────────
        shooter_pos = pos_map.get((int(frames[i]), shooter_id))
        if shooter_pos is None:
            continue
        shooter_x, shooter_y = shooter_pos[0], shooter_pos[1]
        shooter_dist_in = math.hypot(shooter_x - rim_x, shooter_y - rim_y)
        if shooter_dist_in < THREEPT_MIN_INCHES:
            continue

        # ── Arc apex ─────────────────────────────────────────────────────────
        # Find the frame of maximum ball height within the search window.
        # This is the peak of the parabolic arc — purely from position data.
        apex_local = int(np.argmax(ball_z[w]))
        apex_i = i + apex_local
        apex_frame     = int(frames[apex_i])
        apex_game_clock = _safe_clock_str(game_clks[apex_i])
        apex_shot_clock = _safe_clock_str(shot_clks[apex_i])
        apex_ball_z     = round(float(ball_z[apex_i]), 2)

        # ── Nearest Heat defender ────────────────────────────────────────────
        best_id, best_name, best_dist = None, "", float("inf")
        for pid, (name, tid, _) in player_map.items():
            if tid != heat_team_id:
                continue
            dpos = pos_map.get((int(frames[i]), pid))
            if dpos is None:
                continue
            d = math.hypot(dpos[0] - shooter_x, dpos[1] - shooter_y)
            if d < best_dist:
                best_dist = d
                best_id   = pid
                best_name = name
        if best_id is None:
            continue

        # ── Contest features ─────────────────────────────────────────────────
        # Closeout speed: defender position delta over ~500 ms (30 frames at 60 Hz)
        prior_i = max(0, i - 30)
        prior_def = pos_map.get((int(frames[prior_i]), best_id))
        closeout_delta_ft = 0.0
        closeout_speed    = 0.0
        if prior_def is not None:
            prior_dist_in  = math.hypot(prior_def[0] - shooter_x, prior_def[1] - shooter_y)
            closeout_delta_ft = (prior_dist_in - best_dist) / 12.0
            closeout_speed    = closeout_delta_ft / 0.5

        # Contest angle: inline (0°) = defender squarely between shooter and rim
        def_now = pos_map[(int(frames[i]), best_id)]
        angle = _angle_deg(
            shooter_x - def_now[0], shooter_y - def_now[1],  # defender → shooter
            rim_x - shooter_x,      rim_y - shooter_y,        # shooter → rim
        )

        # Hand height: max wrist above defender torso
        lw, rw = def_now[3], def_now[4]
        if not math.isnan(lw) and not math.isnan(rw):
            hand_up_in = max(lw, rw) - def_now[2]
        elif not math.isnan(lw):
            hand_up_in = lw - def_now[2]
        elif not math.isnan(rw):
            hand_up_in = rw - def_now[2]
        else:
            hand_up_in = 0.0

        contest_dist_ft = best_dist / 12.0
        scq = _contest_quality(contest_dist_ft, closeout_speed, angle, hand_up_in)

        # ── PBP outcome ──────────────────────────────────────────────────────
        release_game_clock = _safe_clock_str(game_clks[i])
        release_shot_clock = _safe_clock_str(shot_clks[i])
        period = int(periods[i])
        release_sec = _parquet_clock_to_sec(release_game_clock)

        pbp, pbp_idx = _match_pbp(pbp_events, shooter_id, period, release_sec)
        if pbp:
            matched_pbp_idx.add(pbp_idx)
            pbp_result  = pbp["shot_result"]
            pbp_clock   = _sec_to_mmss(pbp["remaining_sec"])
            pbp_desc    = pbp["description"]
        else:
            pbp_result  = NA
            pbp_clock   = NA
            pbp_desc    = NA

        rows.append({
            # Identifiers
            "game_file":   os.path.basename(path),
            "game_id":     _game_id_from_filename(os.path.basename(path)) or NA,
            "opponent":    opponent,
            "period":      period,
            # Release
            "release_frame":          int(frames[i]),
            "release_game_clock":     release_game_clock,
            "release_shot_clock":     release_shot_clock,
            "shooter_id":             shooter_id,
            "shooter_name":           shooter_name,
            "shooter_team":           shooter_team_name,
            "shooter_dist_to_rim_in": round(shooter_dist_in, 2),
            "rim_x":                  rim_x,
            "rim_y":                  rim_y,
            "release_ball_x":         round(float(ball_x[i]), 3),
            "release_ball_y":         round(float(ball_y[i]), 3),
            "release_ball_z":         round(float(ball_z[i]), 3),
            # Arc apex
            "apex_frame":       apex_frame,
            "apex_game_clock":  apex_game_clock,
            "apex_shot_clock":  apex_shot_clock,
            "apex_ball_z":      apex_ball_z,
            # Contest features
            "min_ball_rim_3d_in":       round(min_rim_dist, 2),
            "nearest_defender_id":      best_id,
            "nearest_defender_name":    best_name,
            "contest_distance_ft":      round(contest_dist_ft, 3),
            "closeout_speed_ft_s":      round(closeout_speed, 3),
            "closeout_delta_ft_500ms":  round(closeout_delta_ft, 3),
            "contest_angle_deg":        round(angle, 2),
            "hand_up_in":               round(hand_up_in, 2),
            "shot_contest_quality":     scq,
            # PBP outcome
            "pbp_shot_result":           pbp_result,
            "pbp_conclusion_game_clock": pbp_clock,
            "pbp_description":           pbp_desc,
        })
        last_shot_idx = i

    return rows, matched_pbp_idx, opponent


# ---------------------------------------------------------------------------
# Unmatched output schema
# ---------------------------------------------------------------------------

UNMATCHED_FIELDNAMES = [
    "unmatched_type",       # "tracking_only" or "pbp_only"
    "game_file", "game_id", "opponent", "period",
    "shooter_id", "shooter_name",
    # Tracking fields — populated for tracking_only, NA for pbp_only
    "release_frame", "release_game_clock", "release_shot_clock",
    "contest_distance_ft", "shot_contest_quality",
    # PBP fields — populated for pbp_only, NA for tracking_only
    "pbp_shot_result", "pbp_conclusion_game_clock", "pbp_description",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified shot contest dataset (Hawk-Eye release + apex + PBP outcome)."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing *_processed.parquet game files.",
    )
    parser.add_argument(
        "--output-csv", default="shot_contest_dataset.csv",
        help="Output CSV path (default: shot_contest_dataset.csv).",
    )
    parser.add_argument(
        "--pbp-delay", type=float, default=0.5,
        help="Seconds to wait between NBA CDN requests (default: 0.5).",
    )
    args = parser.parse_args()

    parquet_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        sys.exit(f"No .parquet files found in {args.input_dir}")

    # Derive unmatched CSV path from the main output path
    base, ext = os.path.splitext(args.output_csv)
    unmatched_csv = f"{base}_unmatched{ext}"

    all_rows:      List[dict] = []
    all_unmatched: List[dict] = []

    # Build a player-id-to-name lookup from tracking data (used to name pbp_only rows)
    player_id_to_name: Dict[int, str] = {}

    for fp in parquet_files:
        basename = os.path.basename(fp)
        game_id  = _game_id_from_filename(basename)
        if not game_id:
            print(f"SKIP (could not parse game ID): {basename}")
            continue

        print(f"\n{basename}  [game {game_id}]")

        print(f"  Fetching PBP ...")
        pbp_actions = _fetch_pbp(game_id)
        pbp_events  = _parse_pbp_opponent_3pa(pbp_actions)
        print(f"  PBP: {len(pbp_events)} opponent 3PA events found")

        game_rows, matched_pbp_idx, opponent = _extract_one_game(fp, pbp_events)
        matched = sum(1 for r in game_rows if r["pbp_shot_result"] != NA)
        print(f"  Tracking: {len(game_rows)} shots detected | PBP matched: {matched} "
              f"| tracking_only: {len(game_rows) - matched} "
              f"| pbp_only: {len(pbp_events) - len(matched_pbp_idx)}")

        all_rows.extend(game_rows)

        # Update name lookup from this game's tracking rows
        for r in game_rows:
            player_id_to_name[r["shooter_id"]] = r["shooter_name"]

        # ── tracking_only: detected by Hawk-Eye but no PBP event matched ──────
        for r in game_rows:
            if r["pbp_shot_result"] != NA:
                continue
            all_unmatched.append({
                "unmatched_type":          "tracking_only",
                "game_file":               r["game_file"],
                "game_id":                 r["game_id"],
                "opponent":                r["opponent"],
                "period":                  r["period"],
                "shooter_id":              r["shooter_id"],
                "shooter_name":            r["shooter_name"],
                "release_frame":           r["release_frame"],
                "release_game_clock":      r["release_game_clock"],
                "release_shot_clock":      r["release_shot_clock"],
                "contest_distance_ft":     r["contest_distance_ft"],
                "shot_contest_quality":    r["shot_contest_quality"],
                "pbp_shot_result":         NA,
                "pbp_conclusion_game_clock": NA,
                "pbp_description":         NA,
            })

        # ── pbp_only: in PBP but no tracking detection matched it ─────────────
        for idx, e in enumerate(pbp_events):
            if idx in matched_pbp_idx:
                continue
            pid = e["person_id"]
            name = player_id_to_name.get(pid) or e["player_abbr"] or str(pid)
            all_unmatched.append({
                "unmatched_type":          "pbp_only",
                "game_file":               basename,
                "game_id":                 game_id,
                "opponent":                opponent,
                "period":                  e["period"],
                "shooter_id":              pid,
                "shooter_name":            name,
                "release_frame":           NA,
                "release_game_clock":      NA,
                "release_shot_clock":      NA,
                "contest_distance_ft":     NA,
                "shot_contest_quality":    NA,
                "pbp_shot_result":         e["shot_result"],
                "pbp_conclusion_game_clock": _sec_to_mmss(e["remaining_sec"]),
                "pbp_description":         e["description"],
            })

        time.sleep(args.pbp_delay)

    # ── Write main dataset ────────────────────────────────────────────────────
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Write unmatched dataset ───────────────────────────────────────────────
    with open(unmatched_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNMATCHED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_unmatched)

    total_matched   = sum(1 for r in all_rows if r["pbp_shot_result"] != NA)
    tracking_only   = sum(1 for r in all_unmatched if r["unmatched_type"] == "tracking_only")
    pbp_only        = sum(1 for r in all_unmatched if r["unmatched_type"] == "pbp_only")

    print(f"\nMain dataset : {len(all_rows)} rows ({total_matched} matched) -> {args.output_csv}")
    print(f"Unmatched    : {len(all_unmatched)} rows "
          f"({tracking_only} tracking_only, {pbp_only} pbp_only) -> {unmatched_csv}")


if __name__ == "__main__":
    main()
