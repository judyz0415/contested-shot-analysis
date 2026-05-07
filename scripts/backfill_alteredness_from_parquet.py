#!/usr/bin/env python3
"""
Backfill `release_path_residual_in` and `alteredness` on an existing shot CSV using Hawk-Eye parquet.

Use when the CSV was built before those columns existed, or to refresh without re-fetching PBP.

Example:
    python scripts/backfill_alteredness_from_parquet.py \\
        --csv data/outputs/shot_contest_dataset.csv \\
        --parquet-dir '/path/to/miami_heat_2025'
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from shot_alteredness import path_residual_and_alteredness  # noqa: E402

RELEASE_PATH_WINDOW_FRAMES = 12
ALTEREDNESS_REF_INCHES = 12.0


def _as_float(v: object) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _ball_arrays_from_parquet(path: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    cols = ["frame", "ball_x", "ball_y", "ball_z"]
    t = pq.read_table(path, columns=cols)
    frame_tbl = t.select(cols).group_by(cols).aggregate([])
    idx = pc.sort_indices(frame_tbl, sort_keys=[("frame", "ascending")])
    frame_tbl = pc.take(frame_tbl, idx)
    frames = [int(x) for x in frame_tbl["frame"].to_pylist()]
    bx = [_as_float(x) for x in frame_tbl["ball_x"].to_pylist()]
    by = [_as_float(x) for x in frame_tbl["ball_y"].to_pylist()]
    bz = [_as_float(x) for x in frame_tbl["ball_z"].to_pylist()]
    return frames, bx, by, bz


def _frame_index(frames: Sequence[int], release_frame: int) -> Optional[int]:
    """First index where frames[i] == release_frame (tracking rows are time-ordered)."""
    for i, f in enumerate(frames):
        if f == release_frame:
            return i
    return None


def _resolve_parquet(parquet_dir: str, game_file: str) -> Optional[str]:
    base = os.path.basename(game_file.strip())
    p = os.path.join(parquet_dir, base)
    if os.path.isfile(p):
        return p
    p2 = os.path.join(parquet_dir, game_file)
    if os.path.isfile(p2):
        return p2
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill path-based alteredness from parquet.")
    parser.add_argument("--csv", required=True, help="Shot dataset CSV to update in place.")
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing *_processed.parquet files (same names as column game_file).",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=RELEASE_PATH_WINDOW_FRAMES,
        help=f"Frames before release used for linear fit (default: {RELEASE_PATH_WINDOW_FRAMES}).",
    )
    parser.add_argument(
        "--alteredness-ref-in",
        type=float,
        default=ALTEREDNESS_REF_INCHES,
        help=f"Inches of 3-D residual mapping to alteredness 100 (default: {ALTEREDNESS_REF_INCHES}).",
    )
    args = parser.parse_args()

    parquet_dir = os.path.expanduser(args.parquet_dir)

    with open(args.csv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            sys.exit("Empty CSV")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    new_cols = ["release_path_residual_in", "alteredness"]
    insert_at = None
    if "shot_contest_quality" in fieldnames:
        insert_at = fieldnames.index("shot_contest_quality") + 1
    else:
        insert_at = len(fieldnames)
    for c in new_cols:
        if c not in fieldnames:
            fieldnames.insert(insert_at, c)
            insert_at += 1

    cache: Dict[str, Tuple[List[int], List[float], List[float], List[float]]] = {}
    failed_loads: Dict[str, str] = {}
    updated = 0
    missing_parquet = 0
    missing_frame = 0

    for row in rows:
        gf = (row.get("game_file") or "").strip()
        rf = (row.get("release_frame") or "").strip()
        if not gf or not rf or rf.upper() == "NA":
            continue
        try:
            release_frame = int(rf)
        except ValueError:
            continue

        pq_path = _resolve_parquet(parquet_dir, gf)
        if not pq_path:
            missing_parquet += 1
            continue

        if pq_path not in cache and pq_path not in failed_loads:
            try:
                cache[pq_path] = _ball_arrays_from_parquet(pq_path)
            except Exception as exc:
                failed_loads[pq_path] = str(exc)

        if pq_path in failed_loads:
            continue

        frames, bx, by, bz = cache[pq_path]
        ri = _frame_index(frames, release_frame)
        if ri is None:
            missing_frame += 1
            continue

        res_in, alt = path_residual_and_alteredness(
            bx,
            by,
            bz,
            ri,
            window_frames=args.window_frames,
            alteredness_reference_inches=args.alteredness_ref_in,
        )
        if math.isfinite(res_in):
            row["release_path_residual_in"] = round(res_in, 3)
        else:
            row["release_path_residual_in"] = "NA"
        if math.isfinite(alt):
            row["alteredness"] = round(alt, 2)
        else:
            row["alteredness"] = "NA"
        updated += 1

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for c in fieldnames:
                row.setdefault(c, "NA")
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Backfill complete: rows updated (tracking path) = {updated}")
    print(f"missing parquet file = {missing_parquet} | release_frame not in parquet = {missing_frame}")
    for p, err in failed_loads.items():
        print(f"FAILED load {p}: {err}")
    print(f"Wrote: {args.csv}")


if __name__ == "__main__":
    main()
