#!/usr/bin/env python3
"""
Build unified shot contest dataset from Hawk-Eye tracking + NBA play-by-play.

One row per opponent 3-point attempt. Columns:

  Identifiers
    game_file, game_id, opponent, period

  Release  (Hawk-Eye: moment ball leaves shooter's hand)
    release_frame, release_game_clock, release_shot_clock
    shooter_id, shooter_name, shooter_team, shooter_2025_26_regular_3pt_pct
    shooter_dist_to_rim_in, rim_x, rim_y
    release_ball_x, release_ball_y, release_ball_z

  Arc apex  (Hawk-Eye: frame of maximum ball height after release)
    apex_frame, apex_game_clock, apex_shot_clock, apex_ball_z

  Contest features  (Hawk-Eye: computed at release frame)
    min_ball_rim_3d_in, min_ball_rim_3d_through_play_in
    nearest_defender_id, nearest_defender_name
    contest_distance_ft, closeout_speed_ft_s, closeout_delta_ft_500ms
    contest_angle_deg, hand_up_in, shot_contest_quality

  PBP outcome  (NBA CDN: matched within 5 s of release, same player+period)
    pbp_shot_result, pbp_conclusion_game_clock, pbp_description
    (all three are "NA" when no PBP event matches)

  QA filters  (automatic tags — tune via CLI constants)
    pbp_rescued — secondary pass anchored on PBP resolution clock + shooter
    analysis_eligible — no if no PBP match, long-range heave, low shot clock, deep release (~half court),
    or ball never approaches the attacking rim within the play window (pass-like)
    exclusion_reason — machine-readable codes (see HEAVE_AUDIT_REASONS subset)
    suspected_low_arc_or_lob — low apex_height heuristic (possible lob mis-tag)
    shooter_2025_26_regular_3pt_pct — shrunk regular-season 3P% (Beta-style; from --player-statistics-csv)
    defender_model_eligible — yes iff row matches lift-model input checks (named defender, finite features)
    defender_model_exclusion_reason — codes when defender_model_eligible=no

Extras:
    <output>_excluded_heaves.csv lists rows flagged by heave / desperation / analytics-geometry rules.

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
import ssl
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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

# PBP-only rescue: PBP clock is resolution time; release is ~0.9–2.7 s earlier on remaining clock.
PBP_RESCUE_LAG_MIN_SEC = 0.9
PBP_RESCUE_LAG_MAX_SEC = 2.8
PBP_RESCUE_PREFERRED_LAG_SEC = 1.55
RELEASE_Z_THRESHOLD_RELAXED = 78.0            # softer crossing for flat / airball releases
RESCUE_RIM_APPROACH_MAX_IN = 38.0           # inches; airballs may not pass 18-in rim proximity check

# Heave / garbage-shot filters (release-time shot clock & shooter distance).
HEAVE_MIN_SHOOTER_DIST_FROM_RIM_IN = 42.0 * 12.0   # default ≥42 ft from rim (~half-court-ish heaves)
MIN_SHOT_CLOCK_FOR_ANALYSIS = 1.0                  # exclude if shot clock strictly below this at release
DESPERATE_SHOT_CLOCK_SEC = 0.8                     # always exclude at or below this (overlaps rule above)

# Heuristic flag when apex never reaches typical jumper height — possible lob / mis-tagged pass.
LOW_ARC_APEX_Z_INCHES = 158.0

# Analytics eligibility (master CSV keeps all rows; these set analysis_eligible=no + exclusion_reason).
# Shooter farther than this from the *attacking* rim (inches) → deep / half-court release, not for contest analytics.
ANALYTICS_MAX_SHOOTER_DIST_TO_RIM_IN = 40.0 * 12.0
# Ball must get at least this close (3D, inches) to the attacking rim at some point from release through the window below.
ANALYTICS_BALL_MUST_APPROACH_RIM_LE_IN = 54.0
# Post-release frames to scan for closest ball–rim approach (passes may miss the rim in the early apex window).
ANALYTICS_PLAY_WINDOW_FRAMES = 360

MIAMI_TEAM_ID = 1610612748
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
NA = "NA"


def _pbp_shot_result_missing(val) -> bool:
    """True when there is no usable PBP shot outcome (CSV may use NA, blank, or float NaN)."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        try:
            if math.isnan(float(val)):
                return True
        except (TypeError, ValueError):
            pass
    s = str(val).strip()
    if not s:
        return True
    if s.upper() == "NA" or s.upper() == "<NA>" or s.lower() in ("nan", "none", "<na>"):
        return True
    return False


# ---------------------------------------------------------------------------
# Output columns (order matters for CSV)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    # Identifiers
    "game_file", "game_id", "opponent", "period",
    # Release
    "release_frame", "release_game_clock", "release_shot_clock",
    "shooter_id", "shooter_name", "shooter_team",
    "shooter_2025_26_regular_3pt_pct",
    "shooter_dist_to_rim_in", "rim_x", "rim_y",
    "release_ball_x", "release_ball_y", "release_ball_z",
    # Arc apex
    "apex_frame", "apex_game_clock", "apex_shot_clock", "apex_ball_z",
    # Contest features
    "min_ball_rim_3d_in",
    "min_ball_rim_3d_through_play_in",
    "nearest_defender_id", "nearest_defender_name",
    "contest_distance_ft", "closeout_speed_ft_s", "closeout_delta_ft_500ms",
    "contest_angle_deg", "hand_up_in", "shot_contest_quality",
    # PBP outcome
    "pbp_shot_result", "pbp_conclusion_game_clock", "pbp_description",
    # QA / filters
    "pbp_rescued",              # yes = recovered from pbp_only via clock-targeted search
    "analysis_eligible",        # yes if row should enter outcome+contest modeling
    "exclusion_reason",         # semicolon-separated codes when not eligible
    "suspected_low_arc_or_lob", # yes if apex height is unusually low for a jumper
    "defender_model_eligible",  # yes = same inclusion rules as scripts/analysis/model_defensive_effectiveness.py
    "defender_model_exclusion_reason",
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


def _min_ball_rim_3d_over_index_range(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    lo: int,
    hi: int,
    rim_x: float,
    rim_y: float,
    rim_z: float,
) -> float:
    """Minimum 3-D distance (inches) from ball to a single rim over frame indices [lo, hi]."""
    lo = max(0, int(lo))
    hi = min(int(hi), len(ball_x) - 1)
    if hi < lo:
        return float("nan")
    dx = ball_x[lo : hi + 1] - rim_x
    dy = ball_y[lo : hi + 1] - rim_y
    dz = ball_z[lo : hi + 1] - rim_z
    return float(np.nanmin(np.sqrt(dx * dx + dy * dy + dz * dz)))


# ---------------------------------------------------------------------------
# PBP: fetch + parse
# ---------------------------------------------------------------------------

def _make_pbp_ssl_context(*, insecure: bool) -> Tuple[ssl.SSLContext, str]:
    """
    Build an SSL context that verifies the NBA CDN.

    Order:
      1) truststore — uses the OS trust store (macOS Keychain, Windows, etc.).
         Fixes common python.org macOS installs where certifi alone still fails.
      2) certifi — Mozilla CA bundle.
      3) ssl.create_default_context() — last resort.
    """
    if insecure:
        return ssl._create_unverified_context(), "disabled verification (INSECURE)"
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "truststore (OS trust store)"
    except Exception:
        pass
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where()), "certifi CA bundle"
    except Exception:
        pass
    return ssl.create_default_context(), "ssl default context"


def _fetch_pbp(game_id: str, *, ssl_context: ssl.SSLContext) -> List[dict]:
    url = PBP_URL.format(game_id=game_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
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


def _release_geometry_ok(i: int, ball_z: np.ndarray, z_thresh: float) -> bool:
    """Upward crossing of z_thresh at i with continuation upward (+2 frames)."""
    return bool(
        ball_z[i - 1] < z_thresh <= ball_z[i]
        and ball_z[i + 2] > ball_z[i]
    )


def _parse_shot_clock_seconds(val) -> float:
    """Shot clock remainder in seconds from parquet string ('18', '0.9', '1:03' rare)."""
    if val is None:
        return float("nan")
    s_raw = _safe_clock_str(val)
    if s_raw == NA:
        return float("nan")
    sec = _parquet_clock_to_sec(s_raw)
    return sec


def _finalize_row_analysis_columns(
    row: dict,
    *,
    heave_min_dist_from_rim_in: float,
    min_shot_clock: float,
    desperate_shot_clock: float,
    analytics_max_shooter_dist_to_rim_in: float,
    analytics_ball_must_approach_rim_le_in: float,
) -> None:
    """Populate pbp_rescued / analysis eligibility / heuristic lob flag."""
    if row.get("pbp_rescued") != "yes":
        row["pbp_rescued"] = "no"

    reasons: List[str] = []
    if _pbp_shot_result_missing(row.get("pbp_shot_result")):
        reasons.append("no_pbp_match")

    sc = _parse_shot_clock_seconds(row.get("release_shot_clock"))
    if not math.isnan(sc):
        # Strict '< 1 s' excludes everything below 1.00 (includes 0.9, 0.8, …).
        if sc + 1e-9 < min_shot_clock:
            reasons.append("shot_clock_below_1s")
        if sc <= desperate_shot_clock + 1e-9:
            reasons.append("desperate_shot_clock_le_0p8")

    try:
        dist_in = float(row["shooter_dist_to_rim_in"])
        if dist_in >= heave_min_dist_from_rim_in - 1e-9:
            reasons.append("heave_long_distance")
        if dist_in >= analytics_max_shooter_dist_to_rim_in - 1e-9:
            reasons.append("shooter_beyond_analytics_arc")
    except (TypeError, ValueError, KeyError):
        pass

    desc = (row.get("pbp_description") or "").lower()
    if "heave" in desc:
        reasons.append("pbp_keyword_heave")

    try:
        play_min = float(row["min_ball_rim_3d_through_play_in"])
        if not math.isnan(play_min) and play_min > analytics_ball_must_approach_rim_le_in + 1e-9:
            reasons.append("ball_far_from_rim_play_window")
    except (TypeError, ValueError, KeyError):
        pass

    row["analysis_eligible"] = "no" if reasons else "yes"
    row["exclusion_reason"] = ";".join(reasons) if reasons else ""

    row["suspected_low_arc_or_lob"] = "no"
    try:
        apex = float(row["apex_ball_z"])
        if apex < LOW_ARC_APEX_Z_INCHES:
            row["suspected_low_arc_or_lob"] = "yes"
    except (TypeError, ValueError, KeyError):
        pass


def _row_float_guard(val) -> float:
    """Parse a CSV/row cell to float; NA / blank -> nan (aligned with analysis scripts)."""
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s or s.upper() == "NA":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _load_shrink_shooter_3pt_priors(
    path: str, *, pseudo_attempts: float = 50.0
) -> Tuple[Dict[int, float], float]:
    """
    Shrunk regular-season 3P% by personId (same logic as model_defensive_effectiveness).
    Returns (person_id -> pct, league_pct).
    """
    tpm: Dict[int, float] = {}
    tpa: Dict[int, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for prow in reader:
            if (prow.get("gameType") or "").strip() != "Regular Season":
                continue
            gd = (prow.get("gameDate") or "").strip()
            if len(gd) >= 10 and gd[:10] < "2025-10-01":
                continue
            pid_raw = (prow.get("personId") or "").strip()
            if not pid_raw:
                continue
            pid = int(pid_raw)
            ta = _row_float_guard(prow.get("threePointersAttempted"))
            tm = _row_float_guard(prow.get("threePointersMade"))
            if not math.isfinite(ta) or not math.isfinite(tm):
                continue
            tpa[pid] = tpa.get(pid, 0.0) + ta
            tpm[pid] = tpm.get(pid, 0.0) + tm

    league_tpa = sum(tpa.values())
    league_tpm = sum(tpm.values())
    league_pct = league_tpm / league_tpa if league_tpa > 1e-9 else 0.36

    priors: Dict[int, float] = {}
    k = pseudo_attempts
    for pid, att in tpa.items():
        made = tpm.get(pid, 0.0)
        priors[pid] = (made + k * league_pct) / (att + k)
    return priors, league_pct


def _assign_shooter_prior_column(row: dict, priors: Dict[int, float], league_pct: float) -> None:
    """Set shooter_2025_26_regular_3pt_pct from priors (league fallback when unknown)."""
    sid = (row.get("shooter_id") or "").strip()
    if not sid:
        row["shooter_2025_26_regular_3pt_pct"] = NA
        return
    try:
        pid = int(sid)
    except ValueError:
        row["shooter_2025_26_regular_3pt_pct"] = NA
        return
    pct = priors.get(pid, league_pct)
    row["shooter_2025_26_regular_3pt_pct"] = f"{pct:.8f}"


def _finalize_defender_model_columns(row: dict) -> None:
    """
    Match row-inclusion rules in scripts/analysis/model_defensive_effectiveness.py:
    analysis_eligible, named nearest defender, parseable PBP make/miss, finite baseline
    + contest numerics (including closeout_delta_ft_500ms), finite release shot clock,
    finite shrunk shooter 3P%.
    """
    reasons: List[str] = []

    if row.get("analysis_eligible") != "yes":
        reasons.append("analysis_ineligible")

    did = (row.get("nearest_defender_id") or "").strip()
    dnm = (row.get("nearest_defender_name") or "").strip()
    if not did or not dnm:
        reasons.append("missing_nearest_defender")

    if not (row.get("shooter_id") or "").strip():
        reasons.append("missing_shooter_id")

    pbp_raw = (row.get("pbp_shot_result") or "").strip().lower()
    if row.get("pbp_shot_result") == NA or not pbp_raw:
        reasons.append("invalid_pbp_outcome")
    elif "made" not in pbp_raw and "miss" not in pbp_raw:
        reasons.append("invalid_pbp_outcome")

    baseline_cols = (
        "shooter_dist_to_rim_in",
        "release_ball_z",
        "apex_ball_z",
        "shooter_2025_26_regular_3pt_pct",
    )
    for c in baseline_cols:
        if not math.isfinite(_row_float_guard(row.get(c))):
            reasons.append(f"nonfinite_{c}")

    sc = _parse_shot_clock_seconds(row.get("release_shot_clock"))
    if not math.isfinite(sc):
        reasons.append("nonfinite_release_shot_clock")

    contest_cols = (
        "contest_distance_ft",
        "closeout_speed_ft_s",
        "closeout_delta_ft_500ms",
        "contest_angle_deg",
        "hand_up_in",
        "shot_contest_quality",
    )
    for c in contest_cols:
        if not math.isfinite(_row_float_guard(row.get(c))):
            reasons.append(f"nonfinite_{c}")

    row["defender_model_eligible"] = "no" if reasons else "yes"
    row["defender_model_exclusion_reason"] = ";".join(dict.fromkeys(reasons)) if reasons else ""


def _build_row_for_release_index(
    i: int,
    path: str,
    frames: np.ndarray,
    periods: np.ndarray,
    game_clks: np.ndarray,
    shot_clks: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    last_touch: np.ndarray,
    player_map: Dict[int, Tuple[str, int, str]],
    pos_map: Dict[Tuple[int, int], Tuple],
    heat_team_id: int,
    opponent: str,
    pbp_events: List[dict],
    *,
    forced_pbp: Optional[Tuple[dict, int]] = None,
    rim_approach_max_in: float = 18.0,
    allowed_z_thresholds: Tuple[float, ...] = (RELEASE_Z_THRESHOLD,),
    pbp_rescued_flag: str = "no",
    analytics_play_window_frames: int = ANALYTICS_PLAY_WINDOW_FRAMES,
) -> Tuple[Optional[dict], int]:
    """
    Build one output row for a candidate release frame index i.
    Returns (row, pbp_idx) with pbp_idx = -1 if no PBP row should be marked used.
    """
    if not any(_release_geometry_ok(i, ball_z, zt) for zt in allowed_z_thresholds):
        return None, -1

    raw_touch = last_touch[i]
    if raw_touch is None or (isinstance(raw_touch, float) and np.isnan(raw_touch)):
        return None, -1
    shooter_id = int(raw_touch)
    shooter_info = player_map.get(shooter_id)
    if shooter_info is None:
        return None, -1
    shooter_name, shooter_team_id, shooter_team_name = shooter_info
    if shooter_team_id == heat_team_id:
        return None, -1

    w = slice(i, i + APEX_SEARCH_FRAMES)
    rim_dists: List[float] = []
    for rx, ry in RIMS_XY:
        d3 = np.sqrt(
            (ball_x[w] - rx) ** 2 + (ball_y[w] - ry) ** 2 + (ball_z[w] - RIM_Z) ** 2
        )
        rim_dists.append(float(np.nanmin(d3)))
    min_rim_dist = min(rim_dists)
    if min_rim_dist > rim_approach_max_in:
        return None, -1
    rim_x, rim_y = RIMS_XY[0 if rim_dists[0] < rim_dists[1] else 1]

    shooter_pos = pos_map.get((int(frames[i]), shooter_id))
    if shooter_pos is None:
        return None, -1
    shooter_x, shooter_y = shooter_pos[0], shooter_pos[1]
    shooter_dist_in = math.hypot(shooter_x - rim_x, shooter_y - rim_y)
    if shooter_dist_in < THREEPT_MIN_INCHES:
        return None, -1

    apex_local = int(np.argmax(ball_z[w]))
    apex_i = i + apex_local
    apex_frame = int(frames[apex_i])
    apex_game_clock = _safe_clock_str(game_clks[apex_i])
    apex_shot_clock = _safe_clock_str(shot_clks[apex_i])
    apex_ball_z = round(float(ball_z[apex_i]), 2)

    play_hi = min(len(frames) - 1, i + int(analytics_play_window_frames))
    min_ball_play = _min_ball_rim_3d_over_index_range(
        ball_x, ball_y, ball_z, i, play_hi, rim_x, rim_y, RIM_Z
    )

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
            best_id = pid
            best_name = name
    if best_id is None:
        return None, -1

    prior_i = max(0, i - 30)
    prior_def = pos_map.get((int(frames[prior_i]), best_id))
    closeout_delta_ft = 0.0
    closeout_speed = 0.0
    if prior_def is not None:
        prior_dist_in = math.hypot(prior_def[0] - shooter_x, prior_def[1] - shooter_y)
        closeout_delta_ft = (prior_dist_in - best_dist) / 12.0
        closeout_speed = closeout_delta_ft / 0.5

    def_now = pos_map[(int(frames[i]), best_id)]
    angle = _angle_deg(
        shooter_x - def_now[0],
        shooter_y - def_now[1],
        rim_x - shooter_x,
        rim_y - shooter_y,
    )

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

    release_game_clock = _safe_clock_str(game_clks[i])
    release_shot_clock = _safe_clock_str(shot_clks[i])
    period = int(periods[i])
    release_sec = _parquet_clock_to_sec(release_game_clock)

    pbp_idx = -1
    if forced_pbp is not None:
        pbp, pbp_idx = forced_pbp
        pbp_result = pbp["shot_result"]
        pbp_clock = _sec_to_mmss(pbp["remaining_sec"])
        pbp_desc = pbp["description"]
    else:
        pbp, pbp_idx = _match_pbp(pbp_events, shooter_id, period, release_sec)
        if pbp:
            pbp_result = pbp["shot_result"]
            pbp_clock = _sec_to_mmss(pbp["remaining_sec"])
            pbp_desc = pbp["description"]
        else:
            pbp_idx = -1
            pbp_result = NA
            pbp_clock = NA
            pbp_desc = NA

    row = {
        "game_file": os.path.basename(path),
        "game_id": _game_id_from_filename(os.path.basename(path)) or NA,
        "opponent": opponent,
        "period": period,
        "release_frame": int(frames[i]),
        "release_game_clock": release_game_clock,
        "release_shot_clock": release_shot_clock,
        "shooter_id": shooter_id,
        "shooter_name": shooter_name,
        "shooter_team": shooter_team_name,
        "shooter_2025_26_regular_3pt_pct": NA,
        "shooter_dist_to_rim_in": round(shooter_dist_in, 2),
        "rim_x": rim_x,
        "rim_y": rim_y,
        "release_ball_x": round(float(ball_x[i]), 3),
        "release_ball_y": round(float(ball_y[i]), 3),
        "release_ball_z": round(float(ball_z[i]), 3),
        "apex_frame": apex_frame,
        "apex_game_clock": apex_game_clock,
        "apex_shot_clock": apex_shot_clock,
        "apex_ball_z": apex_ball_z,
        "min_ball_rim_3d_in": round(min_rim_dist, 2),
        "min_ball_rim_3d_through_play_in": round(float(min_ball_play), 2),
        "nearest_defender_id": best_id,
        "nearest_defender_name": best_name,
        "contest_distance_ft": round(contest_dist_ft, 3),
        "closeout_speed_ft_s": round(closeout_speed, 3),
        "closeout_delta_ft_500ms": round(closeout_delta_ft, 3),
        "contest_angle_deg": round(angle, 2),
        "hand_up_in": round(hand_up_in, 2),
        "shot_contest_quality": scq,
        "pbp_shot_result": pbp_result,
        "pbp_conclusion_game_clock": pbp_clock,
        "pbp_description": pbp_desc,
        "pbp_rescued": pbp_rescued_flag,
        "analysis_eligible": NA,
        "exclusion_reason": NA,
        "suspected_low_arc_or_lob": NA,
        "defender_model_eligible": NA,
        "defender_model_exclusion_reason": NA,
    }
    return row, pbp_idx


def _rescue_unmatched_pbp_shots(
    path: str,
    frames: np.ndarray,
    periods: np.ndarray,
    game_clks: np.ndarray,
    shot_clks: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    last_touch: np.ndarray,
    player_map: Dict[int, Tuple[str, int, str]],
    pos_map: Dict[Tuple[int, int], Tuple],
    heat_team_id: int,
    opponent: str,
    pbp_events: List[dict],
    matched_pbp_idx: set,
    used_release_frames: Set[int],
    lag_min_sec: float,
    lag_max_sec: float,
    preferred_lag_sec: float,
    analytics_play_window_frames: int,
) -> List[dict]:
    """
    Second pass: align unmatched PBP 3PA to tracking using resolution clock + lag window.
    Helps with back-to-back attempts and flat trajectories missed by the primary pass.
    """
    unmatched_idx = [j for j in range(len(pbp_events)) if j not in matched_pbp_idx]
    # Chronological within each period (higher remaining clock = earlier in period)
    unmatched_idx.sort(key=lambda j: (pbp_events[j]["period"], -pbp_events[j]["remaining_sec"]))

    rescued: List[dict] = []
    z_tiers_rescue = (
        RELEASE_Z_THRESHOLD,
        RELEASE_Z_THRESHOLD_RELAXED,
        72.0,
    )

    for j in unmatched_idx:
        evt = pbp_events[j]
        conclusion = evt["remaining_sec"]
        if math.isnan(conclusion):
            continue
        win_lo = conclusion + lag_min_sec
        win_hi = conclusion + lag_max_sec
        candidates: List[int] = []
        for i in range(3, len(frames) - APEX_SEARCH_FRAMES):
            if int(periods[i]) != evt["period"]:
                continue
            lt = last_touch[i]
            if lt is None or (isinstance(lt, float) and np.isnan(lt)):
                continue
            if int(lt) != evt["person_id"]:
                continue
            rframe = int(frames[i])
            if rframe in used_release_frames:
                continue
            gsec = _parquet_clock_to_sec(_safe_clock_str(game_clks[i]))
            if math.isnan(gsec):
                continue
            if not (win_lo <= gsec <= win_hi):
                continue
            candidates.append(i)

        if not candidates:
            continue

        pref = conclusion + preferred_lag_sec
        candidates.sort(
            key=lambda ii: abs(_parquet_clock_to_sec(_safe_clock_str(game_clks[ii])) - pref)
        )

        placed = False
        for i in candidates:
            row, _ = _build_row_for_release_index(
                i,
                path,
                frames,
                periods,
                game_clks,
                shot_clks,
                ball_x,
                ball_y,
                ball_z,
                last_touch,
                player_map,
                pos_map,
                heat_team_id,
                opponent,
                pbp_events,
                forced_pbp=(evt, j),
                rim_approach_max_in=RESCUE_RIM_APPROACH_MAX_IN,
                allowed_z_thresholds=z_tiers_rescue,
                pbp_rescued_flag="yes",
                analytics_play_window_frames=analytics_play_window_frames,
            )
            if row is None:
                continue
            used_release_frames.add(int(frames[i]))
            matched_pbp_idx.add(j)
            rescued.append(row)
            placed = True
            break
        if not placed:
            continue

    return rescued


# ---------------------------------------------------------------------------
# Per-game extraction
# ---------------------------------------------------------------------------

def _extract_one_game(
    path: str,
    pbp_events: List[dict],
    *,
    rescue_pbp_only: bool,
    pbp_rescue_lag_min: float,
    pbp_rescue_lag_max: float,
    pbp_rescue_preferred_lag: float,
    heave_min_dist_from_rim_in: float,
    min_shot_clock_for_analysis: float,
    desperate_shot_clock_sec: float,
    analytics_max_shooter_dist_to_rim_in: float,
    analytics_ball_must_approach_rim_le_in: float,
    analytics_play_window_frames: int,
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

    frames = np.array(frame_tbl["frame"], dtype=np.int64)
    periods = np.array(frame_tbl["period"], dtype=np.int64)
    game_clks = np.array(frame_tbl["gameClockTime"])
    shot_clks = np.array(frame_tbl["shotClockTime"])
    ball_x = np.array(frame_tbl["ball_x"], dtype=float)
    ball_y = np.array(frame_tbl["ball_y"], dtype=float)
    ball_z = np.array(frame_tbl["ball_z"], dtype=float)
    last_touch = np.array(frame_tbl["last_touch_player_id"])

    rows: List[dict] = []
    matched_pbp_idx: set = set()
    used_release_frames: Set[int] = set()
    last_shot_idx = -999

    for i in range(3, len(frames) - APEX_SEARCH_FRAMES):
        if i - last_shot_idx < 20:
            continue

        row, pbp_idx = _build_row_for_release_index(
            i,
            path,
            frames,
            periods,
            game_clks,
            shot_clks,
            ball_x,
            ball_y,
            ball_z,
            last_touch,
            player_map,
            pos_map,
            heat_team_id,
            opponent,
            pbp_events,
            forced_pbp=None,
            rim_approach_max_in=18.0,
            allowed_z_thresholds=(RELEASE_Z_THRESHOLD,),
            pbp_rescued_flag="no",
            analytics_play_window_frames=analytics_play_window_frames,
        )
        if row is None:
            continue
        if pbp_idx >= 0:
            matched_pbp_idx.add(pbp_idx)
        used_release_frames.add(int(frames[i]))
        _finalize_row_analysis_columns(
            row,
            heave_min_dist_from_rim_in=heave_min_dist_from_rim_in,
            min_shot_clock=min_shot_clock_for_analysis,
            desperate_shot_clock=desperate_shot_clock_sec,
            analytics_max_shooter_dist_to_rim_in=analytics_max_shooter_dist_to_rim_in,
            analytics_ball_must_approach_rim_le_in=analytics_ball_must_approach_rim_le_in,
        )
        rows.append(row)
        last_shot_idx = i

    if rescue_pbp_only:
        extra = _rescue_unmatched_pbp_shots(
            path,
            frames,
            periods,
            game_clks,
            shot_clks,
            ball_x,
            ball_y,
            ball_z,
            last_touch,
            player_map,
            pos_map,
            heat_team_id,
            opponent,
            pbp_events,
            matched_pbp_idx,
            used_release_frames,
            lag_min_sec=pbp_rescue_lag_min,
            lag_max_sec=pbp_rescue_lag_max,
            preferred_lag_sec=pbp_rescue_preferred_lag,
            analytics_play_window_frames=analytics_play_window_frames,
        )
        for row in extra:
            _finalize_row_analysis_columns(
                row,
                heave_min_dist_from_rim_in=heave_min_dist_from_rim_in,
                min_shot_clock=min_shot_clock_for_analysis,
                desperate_shot_clock=desperate_shot_clock_sec,
                analytics_max_shooter_dist_to_rim_in=analytics_max_shooter_dist_to_rim_in,
                analytics_ball_must_approach_rim_le_in=analytics_ball_must_approach_rim_le_in,
            )
            rows.append(row)

    return rows, matched_pbp_idx, opponent


# ---------------------------------------------------------------------------
# Unmatched output schema
# ---------------------------------------------------------------------------

# Reasons mirrored in _finalize_row_analysis_columns()
HEAVE_AUDIT_REASONS = frozenset({
    "no_pbp_match",
    "heave_long_distance",
    "shot_clock_below_1s",
    "desperate_shot_clock_le_0p8",
    "pbp_keyword_heave",
    "shooter_beyond_analytics_arc",
    "ball_far_from_rim_play_window",
})

HEAVE_AUDIT_FIELDNAMES = [
    "game_file", "game_id", "opponent", "period",
    "release_frame", "release_game_clock", "release_shot_clock",
    "shooter_id", "shooter_name",
    "shooter_dist_to_rim_in",
    "min_ball_rim_3d_through_play_in",
    "pbp_shot_result", "pbp_conclusion_game_clock", "pbp_description",
    "pbp_rescued", "exclusion_reason",
]


def _row_is_heave_audit(row: dict) -> bool:
    """True if excluded primarily for distance / end-of-clock heave rules."""
    parts = [p for p in (row.get("exclusion_reason") or "").split(";") if p]
    return bool(parts) and any(p in HEAVE_AUDIT_REASONS for p in parts)


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
    parser.add_argument(
        "--insecure-pbp-ssl",
        action="store_true",
        help="Emergency only: skip TLS verification for NBA CDN (not recommended).",
    )
    parser.add_argument(
        "--no-pbp-rescue", action="store_true",
        help="Skip second-pass matching for pbp_only events (clock-guided rescue).",
    )
    parser.add_argument(
        "--pbp-rescue-lag-min", type=float, default=PBP_RESCUE_LAG_MIN_SEC,
        help="Min seconds (release_remaining - conclusion_remaining) for rescue window.",
    )
    parser.add_argument(
        "--pbp-rescue-lag-max", type=float, default=PBP_RESCUE_LAG_MAX_SEC,
        help="Max seconds for rescue window.",
    )
    parser.add_argument(
        "--pbp-rescue-preferred-lag", type=float, default=PBP_RESCUE_PREFERRED_LAG_SEC,
        help="Preferred flight-time offset inside the rescue window.",
    )
    parser.add_argument(
        "--heave-min-ft-from-rim", type=float, default=42.0,
        help="Treat shooter distances ≥ this many feet from attacking rim as heaves.",
    )
    parser.add_argument(
        "--min-shot-clock-analysis", type=float, default=MIN_SHOT_CLOCK_FOR_ANALYSIS,
        help="Exclude attempts with shot clock strictly below this at release.",
    )
    parser.add_argument(
        "--desperate-shot-clock", type=float, default=DESPERATE_SHOT_CLOCK_SEC,
        help="Also tag <= this shot-clock value as desperation (logged in exclusion_reason).",
    )
    parser.add_argument(
        "--analytics-max-shooter-ft-from-rim",
        type=float,
        default=40.0,
        help="analysis_eligible=no if shooter is ≥ this many feet from attacking rim (deep / half-court).",
    )
    parser.add_argument(
        "--analytics-ball-must-get-within-inches-of-rim",
        type=float,
        default=54.0,
        help="analysis_eligible=no if min 3D ball–attacking-rim distance over the post-release window "
        "never reaches this close (inches); filters pass-like trajectories.",
    )
    parser.add_argument(
        "--analytics-play-window-frames",
        type=int,
        default=ANALYTICS_PLAY_WINDOW_FRAMES,
        help="Frames after release to scan for closest ball–rim approach (default ~6 s @ 60 Hz).",
    )
    parser.add_argument(
        "--excluded-heaves-csv", default=None,
        help="Optional path for heave / desperation exclusions audit file "
        "(default: <output-csv basename>_excluded_heaves.csv).",
    )
    parser.add_argument(
        "--player-statistics-csv",
        default="data/PlayerStatistics.csv",
        help="NBA PlayerStatistics-style CSV for shrunk shooter 3P%% priors "
        "(same as model_defensive_effectiveness). If missing, league fallback is used.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(
            f"Input directory does not exist:\n  {args.input_dir}\n"
            "Pass the real folder with your *_processed.parquet files — not a placeholder path."
        )

    parquet_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        sys.exit(f"No .parquet files found in {args.input_dir}")

    base, ext = os.path.splitext(args.output_csv)
    unmatched_csv = f"{base}_unmatched{ext}"
    heaves_csv = args.excluded_heaves_csv or f"{base}_excluded_heaves{ext}"

    heave_dist_in = float(args.heave_min_ft_from_rim) * 12.0
    analytics_max_shooter_in = float(args.analytics_max_shooter_ft_from_rim) * 12.0
    analytics_ball_le_in = float(args.analytics_ball_must_get_within_inches_of_rim)
    analytics_play_frames = int(args.analytics_play_window_frames)
    rescue_enabled = not args.no_pbp_rescue

    pbp_ssl, pbp_ssl_label = _make_pbp_ssl_context(insecure=args.insecure_pbp_ssl)
    print(f"PBP HTTPS verification: {pbp_ssl_label}")

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
        pbp_actions = _fetch_pbp(game_id, ssl_context=pbp_ssl)
        pbp_events  = _parse_pbp_opponent_3pa(pbp_actions)
        print(f"  PBP: {len(pbp_events)} opponent 3PA events found")

        game_rows, matched_pbp_idx, opponent = _extract_one_game(
            fp,
            pbp_events,
            rescue_pbp_only=rescue_enabled,
            pbp_rescue_lag_min=args.pbp_rescue_lag_min,
            pbp_rescue_lag_max=args.pbp_rescue_lag_max,
            pbp_rescue_preferred_lag=args.pbp_rescue_preferred_lag,
            heave_min_dist_from_rim_in=heave_dist_in,
            min_shot_clock_for_analysis=args.min_shot_clock_analysis,
            desperate_shot_clock_sec=args.desperate_shot_clock,
            analytics_max_shooter_dist_to_rim_in=analytics_max_shooter_in,
            analytics_ball_must_approach_rim_le_in=analytics_ball_le_in,
            analytics_play_window_frames=analytics_play_frames,
        )
        matched = sum(1 for r in game_rows if r["pbp_shot_result"] != NA)
        rescued_n = sum(1 for r in game_rows if r.get("pbp_rescued") == "yes")
        eligible_n = sum(1 for r in game_rows if r.get("analysis_eligible") == "yes")
        print(
            f"  Tracking rows: {len(game_rows)} | PBP matched rows: {matched} "
            f"| pbp_rescued: {rescued_n} | analysis_eligible: {eligible_n}\n"
            f"  Unmatched implied: tracking_only {len(game_rows) - matched} | "
            f"pbp_only {len(pbp_events) - len(matched_pbp_idx)}"
        )

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

    stats_path = os.path.abspath(args.player_statistics_csv)
    if os.path.isfile(stats_path):
        shooter_priors, league_3p = _load_shrink_shooter_3pt_priors(stats_path)
        print(f"\nShooter priors loaded: {stats_path} ({len(shooter_priors)} players, league={league_3p:.4f})")
    else:
        shooter_priors, league_3p = {}, 0.36
        print(
            f"\nWARNING: Player statistics not found:\n  {stats_path}\n"
            f"Using empty priors with league_pct={league_3p} for shooter_2025_26_regular_3pt_pct."
        )

    for r in all_rows:
        _assign_shooter_prior_column(r, shooter_priors, league_3p)
        _finalize_defender_model_columns(r)

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

    heave_rows = [{k: r.get(k, NA) for k in HEAVE_AUDIT_FIELDNAMES} for r in all_rows if _row_is_heave_audit(r)]
    with open(heaves_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEAVE_AUDIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(heave_rows)

    total_matched   = sum(1 for r in all_rows if r["pbp_shot_result"] != NA)
    tracking_only   = sum(1 for r in all_unmatched if r["unmatched_type"] == "tracking_only")
    pbp_only        = sum(1 for r in all_unmatched if r["unmatched_type"] == "pbp_only")
    eligible_all    = sum(1 for r in all_rows if r.get("analysis_eligible") == "yes")
    dm_eligible_all = sum(1 for r in all_rows if r.get("defender_model_eligible") == "yes")

    print(f"\nMain dataset : {len(all_rows)} rows ({total_matched} matched) -> {args.output_csv}")
    print(f"Eligible     : {eligible_all} rows (analysis_eligible=yes)")
    print(f"Defender model eligible: {dm_eligible_all} rows (defender_model_eligible=yes)")
    print(f"Excluded heaves audit: {len(heave_rows)} rows -> {heaves_csv}")
    print(f"Unmatched    : {len(all_unmatched)} rows "
          f"({tracking_only} tracking_only, {pbp_only} pbp_only) -> {unmatched_csv}")


if __name__ == "__main__":
    main()
