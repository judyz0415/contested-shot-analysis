"""Match opponent-team 3PA to tracking parquet centroid locations using NBA PBP clocks."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ONEDRIVE = Path(
    "/Users/mariaangellobon/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)

# Miami Heat Franchise ID (stats.nba.com / NBA CDN APIs)
MIAMI_TEAM_ID = 1610612748

# Ball-height local-maximum search (~60 Hz tracking): widen until a peak survives scoring.
ARC_FRAME_WINDOWS = (35, 55, 75, 105, 150, 240, 360)
# Prefer tall peaks tied to `frame_match`; penalise peaks far away (handles dribble bumps).
ARC_DISTANCE_WEIGHT = 2.5


def parquet_filename_to_game_id(name: str) -> str | None:
    """e.g. nba_game_0022500062_processed.parquet -> 0022500062"""
    m = re.search(r"(\d{10})", name)
    return m.group(1) if m else None


def fetch_play_by_play(game_id: str) -> dict:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read())
    except OSError as e:
        raise RuntimeError(f"Failed to download NBA play-by-play for game {game_id}: {url}") from e


def iso_clock_to_remaining_seconds(iso_clock: str) -> float:
    m = re.fullmatch(r"PT(?P<m>\d+)M(?P<s>\d+\.?\d*)S", iso_clock.strip())
    if not m:
        raise ValueError(f"Unrecognized clock: {iso_clock!r}")
    return int(m["m"]) * 60.0 + float(m["s"])


def parquet_clock_to_remaining_seconds(game_clock_time) -> float:
    """Mirror parquet encoding (MM:SS for most time; decimals under ~1 minute)."""
    if pd.isna(game_clock_time):
        return float("nan")
    s = str(game_clock_time).strip()
    if ":" in s:
        mins, secs = s.split(":", 1)
        return int(mins) * 60.0 + float(secs)
    return float(s)


def collect_opponent_three_point_attempts(actions: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, a in enumerate(actions):
        if not a.get("isFieldGoal"):
            continue
        desc = a.get("description") or ""
        if "3PT" not in desc and "3-PT" not in desc:
            continue
        tid = a.get("teamId")
        if tid is None or tid == MIAMI_TEAM_ID:
            continue
        pid = a.get("personId")
        if not pid:
            continue
        clock_iso = a.get("clock") or ""
        out.append(
            {
                "action_index": i,
                "period": int(a["period"]),
                "iso_clock": clock_iso,
                "remaining_sec_target": iso_clock_to_remaining_seconds(clock_iso),
                "person_id": int(pid),
                "shooter_abbr": (a.get("playerNameI") or "").strip(),
                "shot_result": (a.get("shotResult") or "").strip(),
                "opponent_tricode": (a.get("teamTricode") or "").strip(),
                "description": desc.strip(),
                "pbp_timeActual_iso": str(a.get("timeActual") or "").strip(),
            }
        )
    return out


def attach_centroids(
    events: list[dict], parquet_path: Path, max_remain_delta: float = 1.25
) -> pd.DataFrame:
    cols = [
        "period",
        "gameClockTime",
        "player_id",
        "fullName",
        "centroid_x",
        "centroid_y",
        "frame",
        "timeUTC",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["_remain"] = df["gameClockTime"].map(parquet_clock_to_remaining_seconds)

    enriched: list[dict] = []
    for ev in events:
        sub = df[
            (df["period"] == ev["period"])
            & (df["player_id"] == ev["person_id"])
        ]
        row_out = {**ev}
        if sub.empty:
            row_out.update(
                {
                    "tracking_clock_match": "",
                    "centroid_x": float("nan"),
                    "centroid_y": float("nan"),
                    "frame_match": "",
                    "tracking_clock_timeUTC_snap": "",
                    "remaining_sec_actual": float("nan"),
                    "remain_sec_abs_delta": float("nan"),
                    "full_name_roster": "",
                    "tracking_clock_ok": False,
                }
            )
            enriched.append(row_out)
            continue

        deltas = (sub["_remain"] - ev["remaining_sec_target"]).abs()
        j = deltas.idxmin()
        best_delta = float(deltas.loc[j])
        tr = df.loc[j]
        row_out.update(
            {
                "tracking_clock_match": tr["gameClockTime"],
                "centroid_x": float(tr["centroid_x"]),
                "centroid_y": float(tr["centroid_y"]),
                "frame_match": int(tr["frame"]),
                "tracking_clock_timeUTC_snap": _str_or_empty(tr.get("timeUTC")),
                "remaining_sec_actual": float(tr["_remain"]),
                "remain_sec_abs_delta": best_delta,
                "full_name_roster": str(tr["fullName"]),
                "tracking_clock_ok": bool(best_delta <= max_remain_delta),
            }
        )
        enriched.append(row_out)

    return pd.DataFrame(enriched)


def _str_or_empty(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def load_ball_tracking_unique_frames(parquet_path: Path) -> pd.DataFrame:
    """Ball is identical for every skeleton row sharing a frame."""
    tbl = pd.read_parquet(parquet_path, columns=["frame", "timeUTC", "ball_z"], engine="pyarrow")
    tbl = tbl.drop_duplicates("frame", keep="first").sort_values("frame")
    tbl["hawkeye_ts"] = pd.to_datetime(tbl["timeUTC"], utc=True, format="mixed")
    tbl = tbl.reset_index(drop=True)
    return tbl


def _local_maxima_indices(z_values: np.ndarray) -> list[int]:
    out: list[int] = []
    for i in range(1, len(z_values) - 1):
        if z_values[i] > z_values[i - 1] and z_values[i] > z_values[i + 1]:
            out.append(i)
    return out


def hawkeye_arc_apex_from_anchor(
    ball_tbl: pd.DataFrame, anchor_frame: int | float | None
) -> dict[str, str | float | int]:
    """Local maximum of ball_z near the centroid-sync frame (jump-shot arc proxy).

    This does **not** classify makes/misses; it only chooses a short ball trajectory segment
    so you can compare timestamps with play-by-play wiring.
    """
    out: dict[str, str | float | int] = {
        "hawkeye_arc_window_frames": "",
        "hawkeye_arc_peak_frame": "",
        "hawkeye_arc_peak_timeUTC": "",
        "hawkeye_arc_peak_ball_z": float("nan"),
        "hawkeye_arc_score": "",
        "hawkeye_arc_note": "",
    }
    try:
        if anchor_frame is None:
            raise ValueError
        if isinstance(anchor_frame, str) and anchor_frame.strip() == "":
            raise ValueError
        if isinstance(anchor_frame, float) and math.isnan(anchor_frame):
            raise ValueError
        af = int(float(anchor_frame))
    except (TypeError, ValueError):
        out["hawkeye_arc_note"] = "anchor_frame_missing_or_invalid"
        return out
    chosen: tuple[float, int, float, int] | None = None  # (score, frame, peak_z, window)
    for window in ARC_FRAME_WINDOWS:
        wd = ball_tbl[
            (ball_tbl["frame"] >= af - window) & (ball_tbl["frame"] <= af + window)
        ].sort_values("frame")
        if wd.shape[0] < 3:
            continue
        z_vals = wd["ball_z"].to_numpy(dtype=float, copy=False)
        peak_ixs = _local_maxima_indices(z_vals)
        if not peak_ixs:
            continue
        scored: list[tuple[float, int, float, int]] = []
        for ix in peak_ixs:
            frame_i = int(wd["frame"].iloc[ix])
            zi = float(z_vals[ix])
            scored.append(
                (zi - ARC_DISTANCE_WEIGHT * abs(frame_i - af), frame_i, zi, int(window))
            )
        cand = max(scored, key=lambda t: t[0])
        chosen = cand
        break

    if chosen is None:
        out["hawkeye_arc_note"] = "no_ball_z_local_maximum_in_first_search_windows"
        return out

    _, peak_frame, peak_z, used_window = chosen
    utc_raw = ball_tbl.loc[ball_tbl["frame"] == peak_frame, "timeUTC"].iloc[0]
    out["hawkeye_arc_window_frames"] = str(int(used_window))
    out["hawkeye_arc_peak_frame"] = int(peak_frame)
    out["hawkeye_arc_peak_ball_z"] = float(peak_z)
    out["hawkeye_arc_peak_timeUTC"] = str(utc_raw).strip()
    out["hawkeye_arc_score"] = f"{chosen[0]:.6f}"
    out["hawkeye_arc_note"] = "ball_z_local_maximum_scored_near_anchor_frame"
    return out


def enrich_arc_and_time_deltas(df_events: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    """Ball-tracking arc timestamp + wall-clock deltas vs NBA `timeActual`."""
    ball_tbl = load_ball_tracking_unique_frames(parquet_path)
    rows: list[dict] = []
    for row in df_events.to_dict(orient="records"):
        merged = {**row}
        merged.update(hawkeye_arc_apex_from_anchor(ball_tbl, merged.get("frame_match")))
        rows.append(merged)

    out = pd.DataFrame(rows)

    pbp_raw = out["pbp_timeActual_iso"].replace("", pd.NA)
    out["pbp_ts_wallclock"] = pd.to_datetime(pbp_raw, utc=True, format="mixed")

    arc_raw = out["hawkeye_arc_peak_timeUTC"].replace("", pd.NA)
    out["hawkeye_arc_peak_ts"] = pd.to_datetime(arc_raw, utc=True, format="mixed")

    snap_raw = out["tracking_clock_timeUTC_snap"].replace("", pd.NA)
    out["hawkeye_centroid_snap_ts"] = pd.to_datetime(snap_raw, utc=True, format="mixed")

    out["delta_sec_pbp_wallclock_minus_hawkeye_arc_peak"] = (
        out["pbp_ts_wallclock"] - out["hawkeye_arc_peak_ts"]
    ).dt.total_seconds()

    out["delta_sec_pbp_wallclock_minus_hawkeye_centroid_snap"] = (
        out["pbp_ts_wallclock"] - out["hawkeye_centroid_snap_ts"]
    ).dt.total_seconds()

    return out


def add_player_labels(df_events: pd.DataFrame) -> pd.DataFrame:
    def label_row(row: pd.Series) -> str:
        r = str(row.get("full_name_roster") or "").strip()
        if r:
            return r
        a = str(row.get("shooter_abbr") or "").strip()
        if a:
            return a
        return f"person_id {int(row['person_id'])}"

    out = df_events.copy()
    out["_player_label"] = out.apply(label_row, axis=1)
    return out


def summarize_by_shooter(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events.empty:
        return pd.DataFrame(columns=["player", "attempts"])
    labelled = add_player_labels(df_events)
    grp = labelled.groupby("_player_label").size().reset_index(name="attempts")
    return grp.rename(columns={"_player_label": "player"}).sort_values(
        "attempts", ascending=False, ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List opponent-team 3PA with period, game-clock, centroid (x,y); "
            "uses NBA CDN play-by-play + local parquet."
        )
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Path to a single processed parquet (e.g. nba_game_0022500062_processed.parquet). "
        "If omitted and --folder is omitted, picks this file inside --games-folder.",
    )
    parser.add_argument(
        "--games-folder",
        type=Path,
        default=DEFAULT_ONEDRIVE,
        help="Folder holding processed parquet games (default: OneDrive project path).",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default="",
        help="10-digit NBA game id (default: inferred from parquet filename when --parquet is set).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="If set, write the event table.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the event rows (summaries always print).",
    )
    args = parser.parse_args()

    parquet_path = args.parquet
    if parquet_path is None:
        parquet_path = args.games_folder / "nba_game_0022500062_processed.parquet"
    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    game_id = args.game_id.strip() or parquet_filename_to_game_id(parquet_path.name)
    if not game_id:
        raise ValueError(
            "Could not determine game id. Pass --game-id XXXXXXXXXX or "
            "--parquet with standard filename."
        )

    actions = fetch_play_by_play(game_id)["game"]["actions"]
    events = collect_opponent_three_point_attempts(actions)
    df_events = attach_centroids(events, parquet_path)
    df_events = enrich_arc_and_time_deltas(df_events, parquet_path)

    if not args.quiet:
        cols_show = [
            "period",
            "iso_clock",
            "pbp_timeActual_iso",
            "tracking_clock_timeUTC_snap",
            "hawkeye_arc_peak_timeUTC",
            "delta_sec_pbp_wallclock_minus_hawkeye_arc_peak",
            "delta_sec_pbp_wallclock_minus_hawkeye_centroid_snap",
            "tracking_clock_match",
            "opponent_tricode",
            "shooter_abbr",
            "full_name_roster",
            "shot_result",
            "centroid_x",
            "centroid_y",
            "hawkeye_arc_peak_frame",
            "hawkeye_arc_peak_ball_z",
            "remain_sec_abs_delta",
        ]
        avail = [c for c in cols_show if c in df_events.columns]
        print(f"\nGame {game_id} | parquet: {parquet_path.name}\n")
        with pd.option_context(
            "display.max_rows", None, "display.width", 200, "display.max_columns", None
        ):
            print(df_events[avail].to_string(index=False))

        if "remain_sec_abs_delta" in df_events.columns:
            large = df_events.loc[
                df_events["remain_sec_abs_delta"] > 1.25, ["iso_clock", "remain_sec_abs_delta"]
            ]
            if len(large):
                print(
                    "\nWarning: clocks with residual > 1.25s (inspect tracking vs PBP):"
                )
                print(large.to_string(index=False))

        print("\n--- Summary ---")

        dcol = "delta_sec_pbp_wallclock_minus_hawkeye_arc_peak"
        if dcol in df_events.columns and df_events[dcol].notna().any():
            med = df_events[dcol].abs().median()
            mn = df_events[dcol].mean()
            print(
                f"\nMedian |PBP wall-clock − Hawkeye arc-apex UTC| = {float(med):.3f}s "
                f"(mean signed delta Δ={float(mn):+.3f}s where Δ=PBP−arc; negative ⇒ NBA event time before arc apex UTC)"
            )

    summary_df = summarize_by_shooter(df_events)

    print(f"\nTotal opponent three-point attempts: {len(df_events)}")
    print("\nAttempts by opponent player:")
    for _, row in summary_df.iterrows():
        print(f"  {int(row['attempts'])}  {row['player']}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_events.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"\nWrote: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
