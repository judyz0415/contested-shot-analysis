#!/usr/bin/env python3
"""
Render a 3D release-frame snapshot for one 3PA:
- shooter (approximate stick figure from centroid + wrist heights)
- closest defender (centroid-distance in release frame)
- ball location

Direct mode:
  python scripts/plot_release_snapshot_3d.py \
    --parquet "/path/to/nba_game_0022500564_processed.parquet" \
    --release-frame 26617 \
    --shooter-id 1631221 \
    --output "data/outputs/release_snapshot_26617.png"

Dataset-row mode:
  python scripts/plot_release_snapshot_3d.py \
    --dataset-csv "data/outputs/shot_contest_dataset.csv" \
    --row-index 61 \
    --parquet-dir "/path/to/processed_parquets"
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _load_frame_rows(parquet_path: str, release_frame: int) -> list[dict]:
    cols = [
        "frame",
        "gameClockTime",
        "ball_x",
        "ball_y",
        "ball_z",
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
    table = pq.read_table(parquet_path, columns=cols)
    mask = pc.equal(table["frame"], release_frame)
    frame_tbl = pc.filter(table, mask)
    rows = frame_tbl.to_pylist()
    if not rows:
        raise ValueError(f"No rows found for frame={release_frame}")
    return rows


def _first_non_null(rows: list[dict], key: str):
    for r in rows:
        val = r.get(key)
        if val is not None:
            return val
    return None


def _nan_or_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _dedupe_players(rows: list[dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for r in rows:
        pid = r.get("player_id")
        if pid is None:
            continue
        i_pid = int(pid)
        if i_pid not in out:
            out[i_pid] = r
    return out


def _nearest_defender(players: Dict[int, dict], shooter_id: int) -> int:
    shooter = players.get(shooter_id)
    if shooter is None:
        raise ValueError(f"Shooter id {shooter_id} not found in this frame.")

    sx, sy = float(shooter["centroid_x"]), float(shooter["centroid_y"])
    shooter_team_id = int(shooter["team_id"])

    best_id: Optional[int] = None
    best_dist = float("inf")
    for pid, r in players.items():
        if pid == shooter_id:
            continue
        team_id = r.get("team_id")
        if team_id is None or int(team_id) == shooter_team_id:
            continue
        d = math.hypot(float(r["centroid_x"]) - sx, float(r["centroid_y"]) - sy)
        if d < best_dist:
            best_dist = d
            best_id = pid

    if best_id is None:
        raise ValueError("Could not identify a defender in the selected frame.")
    return best_id


def _draw_stick(ax, row: dict, color: str, label: str) -> None:
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    cz = float(row["centroid_z"])
    lwz = _nan_or_float(row.get("lWrist_z"))
    rwz = _nan_or_float(row.get("rWrist_z"))

    # Approximate skeleton anchors (inches) using centroid as torso center.
    head = (cx, cy, cz + 10.0)
    neck = (cx, cy, cz + 6.0)
    hip = (cx, cy, cz - 6.0)
    l_shoulder = (cx - 4.0, cy, cz + 5.0)
    r_shoulder = (cx + 4.0, cy, cz + 5.0)
    l_wrist = (cx - 10.0, cy, lwz if not math.isnan(lwz) else cz + 2.0)
    r_wrist = (cx + 10.0, cy, rwz if not math.isnan(rwz) else cz + 2.0)
    l_foot = (cx - 3.0, cy, 0.0)
    r_foot = (cx + 3.0, cy, 0.0)

    segments = [
        (head, neck),
        (neck, hip),
        (neck, l_shoulder),
        (neck, r_shoulder),
        (l_shoulder, l_wrist),
        (r_shoulder, r_wrist),
        (hip, l_foot),
        (hip, r_foot),
    ]
    for a, b in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=2.0)

    ax.scatter([cx], [cy], [cz], color=color, s=50)
    ax.text(cx, cy, cz + 14.0, label, color=color, fontsize=9)


def _load_dataset_row(dataset_csv: str, row_index: int) -> dict:
    """Read a single data row from shot_contest_dataset.csv (1-based index)."""
    if row_index < 1:
        raise ValueError("--row-index must be >= 1 (1-based, excluding header).")
    with open(dataset_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            if i == row_index:
                return row
    raise ValueError(f"Row index {row_index} is out of range for {dataset_csv}.")


def _resolve_parquet_path(game_file: str, parquet_dir: Optional[str]) -> str:
    """Resolve parquet path from game_file with optional parquet-dir prefix."""
    if os.path.isabs(game_file) and os.path.exists(game_file):
        return game_file
    if parquet_dir:
        candidate = os.path.join(parquet_dir, game_file)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(game_file):
        return game_file
    raise ValueError(
        f"Could not resolve parquet file '{game_file}'. "
        "Pass --parquet-dir pointing to the folder with *_processed.parquet files."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 3D snapshot of shooter, ball, and nearest defender at release frame."
    )
    parser.add_argument("--parquet", help="Path to *_processed.parquet file (direct mode).")
    parser.add_argument("--release-frame", type=int, help="Release frame number (direct mode).")
    parser.add_argument("--shooter-id", type=int, help="Shooter player_id (direct mode).")
    parser.add_argument(
        "--dataset-csv",
        help="Path to shot_contest_dataset.csv (dataset-row mode).",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        help="1-based row number in --dataset-csv (excluding header).",
    )
    parser.add_argument(
        "--parquet-dir",
        help="Folder containing parquet files from game_file column (dataset-row mode).",
    )
    parser.add_argument(
        "--defender-id",
        type=int,
        default=None,
        help="Optional defender player_id. If omitted, nearest defender is used.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    if args.dataset_csv or args.row_index:
        if not (args.dataset_csv and args.row_index):
            raise ValueError("When using dataset-row mode, pass both --dataset-csv and --row-index.")
        row = _load_dataset_row(args.dataset_csv, args.row_index)
        parquet_path = _resolve_parquet_path(str(row["game_file"]), args.parquet_dir)
        release_frame = int(float(row["release_frame"]))
        shooter_id = int(float(row["shooter_id"]))
        output = args.output or f"release_snapshot_row_{args.row_index}.png"
    else:
        if not (args.parquet and args.release_frame and args.shooter_id):
            raise ValueError(
                "Pass either dataset-row mode (--dataset-csv + --row-index) "
                "or direct mode (--parquet + --release-frame + --shooter-id)."
            )
        parquet_path = args.parquet
        release_frame = args.release_frame
        shooter_id = args.shooter_id
        output = args.output or "release_snapshot_3d.png"

    rows = _load_frame_rows(parquet_path, release_frame)
    players = _dedupe_players(rows)

    if shooter_id not in players:
        raise ValueError(f"Shooter id {shooter_id} not present at frame {release_frame}.")

    defender_id = args.defender_id if args.defender_id is not None else _nearest_defender(players, shooter_id)
    if defender_id not in players:
        raise ValueError(f"Defender id {defender_id} not present at frame {release_frame}.")

    shooter = players[shooter_id]
    defender = players[defender_id]

    ball_x = _first_non_null(rows, "ball_x")
    ball_y = _first_non_null(rows, "ball_y")
    ball_z = _first_non_null(rows, "ball_z")
    game_clock = _first_non_null(rows, "gameClockTime")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _draw_stick(ax, shooter, color="#1f77b4", label=f"Shooter: {shooter.get('fullName', shooter_id)}")
    _draw_stick(ax, defender, color="#d62728", label=f"Defender: {defender.get('fullName', defender_id)}")

    ax.scatter([float(ball_x)], [float(ball_y)], [float(ball_z)], color="#ff8c00", s=90, marker="o")
    ax.text(float(ball_x), float(ball_y), float(ball_z) + 8.0, "Ball", color="#ff8c00", fontsize=9)

    ax.set_xlabel("X (inches)")
    ax.set_ylabel("Y (inches)")
    ax.set_zlabel("Z (inches)")
    ax.set_title(
        f"3D Release Snapshot | frame={release_frame} | clock={game_clock}\n"
        f"shooter={shooter.get('fullName', shooter_id)} | defender={defender.get('fullName', defender_id)}"
    )

    xs = [float(shooter["centroid_x"]), float(defender["centroid_x"]), float(ball_x)]
    ys = [float(shooter["centroid_y"]), float(defender["centroid_y"]), float(ball_y)]
    x_mid = (min(xs) + max(xs)) / 2.0
    y_mid = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), 96.0)
    half = span / 2.0 + 36.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(0, max(float(ball_z) + 30.0, 120.0))

    out_path = Path(output)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved 3D snapshot to: {out_path}")


if __name__ == "__main__":
    main()
