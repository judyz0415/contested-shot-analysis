#!/usr/bin/env python3
"""
Shot-centric visualization pipeline using shot_contest_dataset.csv + game parquet.

For each selected shot:
1) Save a static 3D release PNG with court lines.
2) Save an interactive rotatable 3D release HTML.
3) Save an interactive rotatable frame-by-frame HTML (pre-release to release).

This script keeps full Hawk-Eye joint coordinates (as available in parquet) and
reuses the animation style from scripts/viz.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from viz import create_plotly_anim, limb_names


BALL_COLS = ["ball_x", "ball_y", "ball_z"]
BASE_COLS = ["frame", "player_id", "fullName", "team_id", "timeUTC"]


def _circle_equation(y: float, inv: bool = True) -> float:
    if inv:
        return ((-168) / 69696) * (y ** 2) - 234
    return ((168) / 69696) * (y ** 2) + 234


def _draw_court_lines(ax) -> None:
    """Minimal half/full-court guide lines in Hawk-Eye coordinates (inches)."""
    def line(p1, p2, lw=1.0):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color="black", linewidth=lw, alpha=0.6)

    # Outer boundary
    outer = [(-564, -300), (-564, 300), (564, 300), (564, -300), (-564, -300)]
    for i in range(len(outer) - 1):
        line(outer[i], outer[i + 1], lw=1.2)

    # Midcourt
    line((0, -300), (0, 300), lw=1.0)

    # 3PT side lines
    line((-564, -264), (-402, -264))
    line((-564, 264), (-402, 264))
    line((564, -264), (402, -264))
    line((564, 264), (402, 264))

    # 3PT arcs (as in viz.py helper)
    ys = list(range(-264, 265, 2))
    x_left = [_circle_equation(y, inv=True) for y in ys]
    x_right = [_circle_equation(y, inv=False) for y in ys]
    ax.plot(x_left, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)
    ax.plot(x_right, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)

    # Paint boxes
    line((-564, -72), (-342, -72))
    line((-342, -72), (-342, 72))
    line((-342, 72), (-564, 72))
    line((564, -72), (342, -72))
    line((342, -72), (342, 72))
    line((342, 72), (564, 72))


def _joint_candidates(joint: str) -> List[str]:
    """
    Return candidate joint base names in common conventions.
    Example: lShoulder -> lShoulder / l_shoulder.
    """
    snake = []
    for i, ch in enumerate(joint):
        if ch.isupper() and i > 0:
            snake.append("_")
        snake.append(ch.lower())
    snake_name = "".join(snake)
    return [joint, snake_name]


def _coord_col(df: pd.DataFrame, joint: str, axis: str) -> Optional[str]:
    for base in _joint_candidates(joint):
        c = f"{base}_{axis}"
        if c in df.columns:
            return c
    return None


def _available_joint_links(df: pd.DataFrame) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _, (j1, j2) in limb_names.items():
        if (
            _coord_col(df, j1, "x") and _coord_col(df, j1, "y") and _coord_col(df, j1, "z")
            and _coord_col(df, j2, "x") and _coord_col(df, j2, "y") and _coord_col(df, j2, "z")
        ):
            out.append((j1, j2))
    return out


def _as_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _resolve_parquet_path(game_file: str, parquet_dir: str) -> str:
    fp = os.path.join(parquet_dir, game_file)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Could not find parquet file: {fp}")
    return fp


def _primary_xy_row(df_frame: pd.DataFrame, pid: int) -> pd.Series:
    p = df_frame[df_frame["player_id"] == pid]
    if p.empty:
        raise ValueError(f"Player {pid} not found in frame {int(df_frame['frame'].iloc[0])}")
    row = p.iloc[0]
    return row


def _xy_for_row(row: pd.Series) -> Tuple[float, float]:
    if "centroid_x" in row.index and "centroid_y" in row.index:
        return _as_float(row["centroid_x"]), _as_float(row["centroid_y"])
    # fallback to torso proxy
    mx = None
    my = None
    for base in ("midHip", "mid_hip", "neck"):
        x = f"{base}_x"
        y = f"{base}_y"
        if x in row.index and y in row.index:
            mx, my = _as_float(row[x]), _as_float(row[y])
            break
    if mx is None or my is None:
        raise ValueError("No centroid_x/centroid_y or torso fallback columns found.")
    return mx, my


def _nearest_defender_id(df_frame: pd.DataFrame, shooter_id: int) -> int:
    shooter = _primary_xy_row(df_frame, shooter_id)
    shooter_team = int(shooter["team_id"])
    sx, sy = _xy_for_row(shooter)

    opp = df_frame[(df_frame["player_id"] != shooter_id) & (df_frame["team_id"] != shooter_team)]
    if opp.empty:
        raise ValueError("No opposing defenders found in release frame.")

    best_pid = None
    best_d = float("inf")
    for _, r in opp.iterrows():
        px, py = _xy_for_row(r)
        d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_pid = int(r["player_id"])
    if best_pid is None:
        raise ValueError("Could not identify nearest defender.")
    return best_pid


def _select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [c for c in BASE_COLS + BALL_COLS if c in df.columns]
    # Keep all available joint xyz columns referenced by viz.limb_names.
    for _, (j1, j2) in limb_names.items():
        for j in (j1, j2):
            for axis in ("x", "y", "z"):
                cc = _coord_col(df, j, axis)
                if cc and cc not in wanted:
                    wanted.append(cc)
    # Useful if present.
    for extra in ("centroid_x", "centroid_y", "centroid_z", "gameClockTime", "period", "shotClockTime"):
        if extra in df.columns and extra not in wanted:
            wanted.append(extra)
    return df[wanted].copy()


def _draw_release_snapshot(
    frame_df: pd.DataFrame,
    shooter_id: int,
    defender_id: int,
    out_png: str,
    title: str,
) -> None:
    links = _available_joint_links(frame_df)
    if not links:
        raise ValueError("No skeleton links available; joint columns missing in frame data.")

    shooter = frame_df[frame_df["player_id"] == shooter_id].iloc[0]
    defender = frame_df[frame_df["player_id"] == defender_id].iloc[0]
    ball = frame_df.iloc[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def plot_player(row: pd.Series, color: str, label: str) -> None:
        for j1, j2 in links:
            x1 = _as_float(row[_coord_col(frame_df, j1, "x")])
            y1 = _as_float(row[_coord_col(frame_df, j1, "y")])
            z1 = _as_float(row[_coord_col(frame_df, j1, "z")])
            x2 = _as_float(row[_coord_col(frame_df, j2, "x")])
            y2 = _as_float(row[_coord_col(frame_df, j2, "y")])
            z2 = _as_float(row[_coord_col(frame_df, j2, "z")])
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=2.0)
        # label near neck or centroid fallback
        if _coord_col(frame_df, "neck", "x"):
            lx = _as_float(row[_coord_col(frame_df, "neck", "x")])
            ly = _as_float(row[_coord_col(frame_df, "neck", "y")])
            lz = _as_float(row[_coord_col(frame_df, "neck", "z")]) + 12.0
        else:
            lx = _as_float(row.get("centroid_x", 0))
            ly = _as_float(row.get("centroid_y", 0))
            lz = _as_float(row.get("centroid_z", 48)) + 12.0
        ax.text(lx, ly, lz, label, color=color, fontsize=9)

    plot_player(shooter, "#1f77b4", f"Shooter: {shooter.get('fullName', shooter_id)}")
    plot_player(defender, "#d62728", f"Defender: {defender.get('fullName', defender_id)}")

    if all(c in frame_df.columns for c in BALL_COLS):
        bx = _as_float(ball["ball_x"])
        by = _as_float(ball["ball_y"])
        bz = _as_float(ball["ball_z"])
        ax.scatter([bx], [by], [bz], color="#ff8c00", s=90)
        ax.text(bx, by, bz + 6.0, "Ball", color="#ff8c00", fontsize=9)

    _draw_court_lines(ax)
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=112)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def _normalize_joint_columns_for_viz(df: pd.DataFrame) -> pd.DataFrame:
    """
    create_plotly_anim expects camelCase names from viz.limb_names.
    If snake_case exists, mirror them into camelCase columns.
    """
    out = df.copy()
    for _, (joint, _) in limb_names.items():
        for axis in ("x", "y", "z"):
            camel = f"{joint}_{axis}"
            if camel in out.columns:
                continue
            src = _coord_col(out, joint, axis)
            if src and src in out.columns:
                out[camel] = out[src]
    # second joint per link
    for _, (_, joint) in limb_names.items():
        for axis in ("x", "y", "z"):
            camel = f"{joint}_{axis}"
            if camel in out.columns:
                continue
            src = _coord_col(out, joint, axis)
            if src and src in out.columns:
                out[camel] = out[src]
    return out


def process_one_shot(
    shot_row: pd.Series,
    parquet_dir: str,
    output_dir: str,
    pre_frames: int,
    post_frames: int,
) -> Dict[str, str]:
    game_file = str(shot_row["game_file"])
    game_id = str(shot_row["game_id"])
    release_frame = int(float(shot_row["release_frame"]))
    shooter_id = int(float(shot_row["shooter_id"]))
    shooter_name = str(shot_row.get("shooter_name", shooter_id))

    parquet_path = _resolve_parquet_path(game_file, parquet_dir)
    game_df = pd.read_parquet(parquet_path)
    game_df = _select_required_columns(game_df)

    release_df = game_df[game_df["frame"] == release_frame].copy()
    if release_df.empty:
        raise ValueError(f"No rows found at release frame {release_frame} for {game_file}.")

    defender_id = _nearest_defender_id(release_df, shooter_id)
    frame_lo = release_frame - pre_frames
    frame_hi_pre_release = release_frame
    frame_hi_with_post = release_frame + post_frames

    release_only = game_df[game_df["frame"] == release_frame].copy()
    release_only = release_only[release_only["player_id"].isin([shooter_id, defender_id])].copy()

    pre_release_window = game_df[(game_df["frame"] >= frame_lo) & (game_df["frame"] <= frame_hi_pre_release)].copy()
    pre_release_window = pre_release_window[pre_release_window["player_id"].isin([shooter_id, defender_id])].copy()
    if pre_release_window.empty:
        raise ValueError("No rows in pre-release frame window for shooter/defender.")

    pre_release_window = _normalize_joint_columns_for_viz(pre_release_window)
    release_only = _normalize_joint_columns_for_viz(release_only)

    # Optional legacy window with post-release frames (kept for flexibility).
    full_window = game_df[(game_df["frame"] >= frame_lo) & (game_df["frame"] <= frame_hi_with_post)].copy()
    full_window = full_window[full_window["player_id"].isin([shooter_id, defender_id])].copy()
    full_window = _normalize_joint_columns_for_viz(full_window)

    stem = f"{game_id}_f{release_frame}_s{shooter_id}_d{defender_id}"
    shot_out_dir = Path(output_dir) / stem
    shot_out_dir.mkdir(parents=True, exist_ok=True)

    png_path = str(shot_out_dir / "release_snapshot_court.png")
    release_html_path = str(shot_out_dir / "release_interactive.html")
    pre_release_html_path = str(shot_out_dir / "pre_release_animation.html")
    full_window_html_path = str(shot_out_dir / "window_animation.html")

    title = (
        f"Release snapshot | game {game_id} | frame {release_frame} | "
        f"{shooter_name} vs defender {defender_id}"
    )
    _draw_release_snapshot(release_df, shooter_id, defender_id, png_path, title)

    release_title = f"Release only | game {game_id} | frame {release_frame} | shooter {shooter_name}"
    fig_release = create_plotly_anim(release_only, ball_column="ball", title=release_title)
    fig_release.write_html(release_html_path, include_plotlyjs="cdn")

    pre_anim_title = (
        f"Pre-release window ({pre_frames} frames before -> release) | "
        f"game {game_id} | shooter {shooter_name}"
    )
    fig_pre = create_plotly_anim(pre_release_window, ball_column="ball", title=pre_anim_title)
    fig_pre.write_html(pre_release_html_path, include_plotlyjs="cdn")

    # Also keep previous full-window output (release + small post tail).
    full_title = (
        f"Shot window ({pre_frames} pre, {post_frames} post) | game {game_id} | "
        f"frame {release_frame} | shooter {shooter_name}"
    )
    fig_full = create_plotly_anim(full_window, ball_column="ball", title=full_title)
    fig_full.write_html(full_window_html_path, include_plotlyjs="cdn")

    return {
        "game_id": game_id,
        "release_frame": str(release_frame),
        "shooter_id": str(shooter_id),
        "defender_id": str(defender_id),
        "release_png": png_path,
        "release_html": release_html_path,
        "pre_release_html": pre_release_html_path,
        "window_html": full_window_html_path,
    }


def _iter_selected_rows(df: pd.DataFrame, row_indices: Optional[Iterable[int]], limit: Optional[int]) -> pd.DataFrame:
    if row_indices:
        idx = [i for i in row_indices if 1 <= i <= len(df)]
        if not idx:
            raise ValueError("No valid --row-indices found within dataset bounds.")
        out = df.iloc[[i - 1 for i in idx]].copy()
    else:
        out = df.copy()
    if limit is not None:
        out = out.head(limit)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate release PNG + interactive release + pre-release animation from shot dataset."
    )
    parser.add_argument(
        "--shots-csv",
        default="data/outputs/shot_contest_dataset.csv",
        help="Shot dataset CSV containing game_file, game_id, release_frame, shooter_id.",
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing *_processed.parquet files referenced by shots CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/shot_visualizations",
        help="Output folder for per-shot visualization artifacts.",
    )
    parser.add_argument(
        "--row-indices",
        default="",
        help="Comma-separated 1-based row indices (e.g. 1,2,150). Empty means all rows.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of shots to process.")
    parser.add_argument("--pre-frames", type=int, default=120, help="Frames before release (~2 sec at 60Hz).")
    parser.add_argument("--post-frames", type=int, default=6, help="Frames after release.")
    args = parser.parse_args()

    shots = pd.read_csv(args.shots_csv)
    required = {"game_file", "game_id", "release_frame", "shooter_id"}
    missing = required - set(shots.columns)
    if missing:
        raise ValueError(f"shots CSV is missing required columns: {sorted(missing)}")

    indices = [int(x.strip()) for x in args.row_indices.split(",") if x.strip()] if args.row_indices else None
    selected = _iter_selected_rows(shots, indices, args.limit)

    records = []
    for _, row in selected.iterrows():
        try:
            rec = process_one_shot(
                row,
                parquet_dir=args.parquet_dir,
                output_dir=args.output_dir,
                pre_frames=args.pre_frames,
                post_frames=args.post_frames,
            )
            records.append(rec)
            print(
                f"DONE game={rec['game_id']} frame={rec['release_frame']} "
                f"shooter={rec['shooter_id']} defender={rec['defender_id']}"
            )
        except Exception as exc:
            print(
                f"FAIL game={row.get('game_id')} frame={row.get('release_frame')} "
                f"shooter={row.get('shooter_id')} :: {exc}"
            )

    summary_path = Path(args.output_dir) / "summary.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
