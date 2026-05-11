#!/usr/bin/env python3
"""
Shot-centric visualization pipeline using shot_contest_dataset.csv + game parquet.

For each selected shot:
1) Save a static 3D release PNG with court lines.
2) Save an interactive rotatable 3D release HTML.
3) Save interactive frame-by-frame animations (pre-release; optional full trace through PBP outcome clock).

This script keeps full Hawk-Eye joint coordinates (as available in parquet) and
reuses the animation style from scripts/viz.py.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from viz import create_plotly_anim, limb_names


BALL_COLS = ["ball_x", "ball_y", "ball_z"]
BASE_COLS = ["frame", "player_id", "fullName", "team_id", "timeUTC"]

# Ceiling on how far past release we chase when clock/CSV inference is capped (~15 s @ 60 Hz).
FULL_WINDOW_FALLBACK_POST_FRAMES = 900
# Tail when CSV has neither outcome_last_frame nor a usable conclusion clock (~3 s @ 60 Hz).
OUTCOME_TAIL_FRAMES_DEFAULT = 180


def _parquet_remaining_seconds(clock_val) -> float:
    """
    Remaining quarter seconds from parquet gameClockTime (matches pipeline semantics).

    Stored as MM:SS for most of the quarter, or a plain float for the final minute.
    Returns NaN if unparseable.
    """
    if clock_val is None:
        return float("nan")
    s = str(clock_val).strip()
    if not s or s.lower() in ("none", "nan", "nat", "na"):
        return float("nan")
    if ":" in s:
        parts = s.split(":", 1)
        try:
            mins = int(parts[0].strip())
            secs = float(parts[1].strip())
            return float(mins * 60 + secs)
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _csv_na(val) -> bool:
    """True when a shot_row cell is missing outcome / convention NA."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip()
    return not s or s.upper() == "NA" or s.lower() == "nan"


def full_window_last_frame_from_shot_row(
    game_df: pd.DataFrame,
    shot_row: pd.Series,
    release_frame: int,
    *,
    post_frames_fallback: int,
    max_extra_frames_after_release: int = FULL_WINDOW_FALLBACK_POST_FRAMES,
    clock_tolerance_sec: float = 3.0,
) -> Tuple[int, str]:
    """
    Last frame for Plotly ``full_window`` tail (make/miss region).

    Priority:
      1) ``shot_row['outcome_last_frame']`` from shot_contest CSV (written by unified pipeline).
      2) Map ``pbp_conclusion_game_clock`` to parquet ``gameClockTime``.
      3) ``release_frame + OUTCOME_TAIL_FRAMES_DEFAULT`` (~3 s @ 60 Hz).

    ``post_frames_fallback`` is legacy (notebook ``POST_FRAMES``); outcomes no longer hinge on it
    unless both CSV timing fields are NA (then falls back same as clock-missing path).

    Returns (frame_hi_int, subtitle_for_title).
    """
    after_rel = game_df[game_df["frame"] >= release_frame]
    max_parquet_f = int(after_rel["frame"].max()) if not after_rel.empty else release_frame

    fb_clock_na = min(release_frame + OUTCOME_TAIL_FRAMES_DEFAULT, max_parquet_f)
    subtitle_na = "~3 s after release (~180 frames @ 60 Hz); no usable outcome timing in CSV"

    # 1) Pipeline column (preferred once dataset is rebuilt)
    ol_raw = shot_row.get("outcome_last_frame")
    if not _csv_na(ol_raw):
        try:
            fh = int(float(str(ol_raw).strip()))
            fh = max(release_frame, min(fh, max_parquet_f, release_frame + max_extra_frames_after_release))
            return fh, f"CSV outcome_last_frame → {fh}"
        except (TypeError, ValueError):
            pass

    # 2) PBP conclusion game clock ↔ parquet clocks
    clk_col = shot_row.get("pbp_conclusion_game_clock")
    target_rem = _parquet_remaining_seconds(clk_col)
    usable_clock = not _csv_na(clk_col) and isinstance(target_rem, float) and math.isfinite(target_rem)

    if not usable_clock:
        fb_legacy = min(
            release_frame + int(post_frames_fallback),
            max_parquet_f,
            release_frame + max_extra_frames_after_release,
        )
        fb_alt = max(fb_clock_na, fb_legacy)
        return fb_alt, subtitle_na + f" | legacy POST_FRAMES floor → {fb_alt}"

    cols = ["frame", "gameClockTime"]
    if "period" in game_df.columns:
        cols.append("period")
    clock_tbl = game_df.loc[:, cols].drop_duplicates(subset=["frame"], keep="first").sort_values("frame")
    clock_tbl = clock_tbl[clock_tbl["frame"] >= release_frame]

    pd_raw = shot_row.get("period")
    if not _csv_na(pd_raw) and "period" in clock_tbl.columns:
        try:
            pd_i = int(float(pd_raw))
            clock_tbl = clock_tbl[clock_tbl["period"] == pd_i]
        except (TypeError, ValueError):
            pass

    if clock_tbl.empty:
        return fb_clock_na, subtitle_na

    sec_rem = clock_tbl["gameClockTime"].apply(_parquet_remaining_seconds)
    clk = clock_tbl.assign(sec_rem=sec_rem).dropna(subset=["sec_rem"])
    if clk.empty:
        return fb_clock_na, subtitle_na

    passed = clk[clk["sec_rem"] <= target_rem + clock_tolerance_sec]
    if passed.empty:
        j = int((clk["sec_rem"] - target_rem).abs().to_numpy().argmin())
        fh = int(clk.iloc[j]["frame"])
        subtitle = f"PBP clock {clk_col!r} → nearest frame {fh}"
    else:
        fh = int(passed["frame"].max())
        subtitle = f"PBP clock {clk_col!r} → last frame ≤ outcome time (frame {fh})"

    fh = max(release_frame, fh)
    cap = min(release_frame + max_extra_frames_after_release, max_parquet_f)
    fh = min(fh, cap)
    return max(min(int(fh), max_parquet_f), release_frame), subtitle


def pbp_result_title_fragment(shot_row: pd.Series) -> str:
    """Short phrase for Plotly titles (make/miss from PBP CSV)."""
    raw = shot_row.get("pbp_shot_result")
    if _csv_na(raw):
        return "Outcome: unknown (no PBP match)"
    s = str(raw).strip()
    upper = s.upper()
    if upper in ("MISS", "MISSED", "MISSING"):
        return "Outcome: miss"
    if upper in ("MADE", "MAKE"):
        return "Outcome: make"
    low = s.lower()
    if "miss" in low:
        return "Outcome: miss"
    if "made" in low or "good" in low:
        return "Outcome: make"
    return f"Outcome: {s}"


def _circle_equation(y: float, inv: bool = True) -> float:
    if inv:
        return ((-168) / 69696) * (y ** 2) - 234
    return ((168) / 69696) * (y ** 2) + 234


def _draw_court_lines(ax) -> None:
    """Minimal full-court guide lines in Hawk-Eye coordinates (inches)."""
    _draw_half_court_lines(ax, "full")


def _draw_half_court_lines(ax, half: str) -> None:
    """
    Floor guide lines in Hawk-Eye inches. `half` is "left" (x<=0), "right" (x>=0), or "full".
    Midcourt is x=0; baskets at x≈±564 in this model.
    """
    def line(p1, p2, lw=1.0):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color="black", linewidth=lw, alpha=0.6)

    if half == "full":
        outer = [(-564, -300), (-564, 300), (564, 300), (564, -300), (-564, -300)]
        for i in range(len(outer) - 1):
            line(outer[i], outer[i + 1], lw=1.2)
        line((0, -300), (0, 300), lw=1.0)
        line((-564, -264), (-402, -264))
        line((-564, 264), (-402, 264))
        line((564, -264), (402, -264))
        line((564, 264), (402, 264))
        ys = list(range(-264, 265, 2))
        x_left = [_circle_equation(y, inv=True) for y in ys]
        x_right = [_circle_equation(y, inv=False) for y in ys]
        ax.plot(x_left, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)
        ax.plot(x_right, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)
        line((-564, -72), (-342, -72))
        line((-342, -72), (-342, 72))
        line((-342, 72), (-564, 72))
        line((564, -72), (342, -72))
        line((342, -72), (342, 72))
        line((342, 72), (564, 72))
        return

    # Single half: boundary = baseline + sidelines + midcourt segment
    if half == "left":
        outer = [(-564, -300), (-564, 300), (0, 300), (0, -300), (-564, -300)]
        for i in range(len(outer) - 1):
            line(outer[i], outer[i + 1], lw=1.2)
        line((0, -300), (0, 300), lw=1.0)
        line((-564, -264), (-402, -264))
        line((-564, 264), (-402, 264))
        ys = list(range(-264, 265, 2))
        x_left = [_circle_equation(y, inv=True) for y in ys]
        ax.plot(x_left, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)
        line((-564, -72), (-342, -72))
        line((-342, -72), (-342, 72))
        line((-342, 72), (-564, 72))
        return

    if half == "right":
        outer = [(0, -300), (564, -300), (564, 300), (0, 300), (0, -300)]
        for i in range(len(outer) - 1):
            line(outer[i], outer[i + 1], lw=1.2)
        line((0, -300), (0, 300), lw=1.0)
        line((564, -264), (402, -264))
        line((564, 264), (402, 264))
        ys = list(range(-264, 265, 2))
        x_right = [_circle_equation(y, inv=False) for y in ys]
        ax.plot(x_right, ys, [0] * len(ys), color="black", linewidth=1.0, alpha=0.6)
        line((564, -72), (342, -72))
        line((342, -72), (342, 72))
        line((342, 72), (564, 72))
        return

    raise ValueError(f"half must be 'left', 'right', or 'full', got {half!r}")


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


def _normalize_nba_game_id(raw) -> str:
    """10-digit NBA game id string (pads leading zeros if CSV/read_csv dropped them)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    digits = "".join(ch for ch in str(raw).strip() if ch.isdigit())
    if not digits:
        return ""
    if len(digits) >= 10:
        return digits[-10:]
    return digits.zfill(10)


def _game_id_from_filename(name: str) -> str:
    m = re.search(r"(\d{10})", name or "")
    return m.group(1) if m else ""


def _resolve_parquet_path(parquet_dir: str, game_file: str, game_id_raw) -> str:
    if not parquet_dir or not str(parquet_dir).strip():
        raise ValueError(
            "--parquet-dir is empty. Export it in the same shell before running:\n"
            '  PARQUET_DIR="/path/to/miami_heat_2025" python ...'
        )
    root = os.path.abspath(os.path.expanduser(str(parquet_dir).strip()))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--parquet-dir is not a directory: {root}")

    candidates: List[str] = []
    gf = str(game_file).strip() if game_file is not None else ""
    if gf and gf.lower() != "nan":
        candidates.append(os.path.join(root, gf))
        candidates.append(os.path.join(root, os.path.basename(gf)))

    nid = _normalize_nba_game_id(game_id_raw)
    if not nid and gf:
        nid = _game_id_from_filename(gf)
    if nid:
        candidates.append(os.path.join(root, f"nba_game_{nid}_processed.parquet"))

    tried: List[str] = []
    for fp in candidates:
        if not fp or fp in tried:
            continue
        tried.append(fp)
        if os.path.isfile(fp):
            return fp

    sample = sorted(glob.glob(os.path.join(root, "nba_game_*_processed.parquet")))[:8]
    sample_txt = "\n  ".join(sample) if sample else "(none matching nba_game_*_processed.parquet)"
    raise FileNotFoundError(
        "Could not find parquet for this shot.\n"
        f"Tried:\n  " + "\n  ".join(tried[:6]) +
        (f"\n...\n(and {len(tried) - 6} more)" if len(tried) > 6 else "") +
        f"\n--parquet-dir: {root}\n"
        f"Sample files in folder:\n  {sample_txt}"
    )


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


def _court_axis_limits() -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Outer court bounds used for axis padding (matches viz.py Hawk-Eye lines, inches)."""
    return (-584.0, 584.0), (-320.0, 320.0), (-5.0, 175.0)


def _mean_action_xy(
    shooter: pd.Series,
    defender: pd.Series,
    ball_row: pd.Series,
    *,
    ball_columns_ok: bool,
) -> Tuple[float, float]:
    xs = [_xy_for_row(shooter)[0], _xy_for_row(defender)[0]]
    ys = [_xy_for_row(shooter)[1], _xy_for_row(defender)[1]]
    if ball_columns_ok:
        xs.append(_as_float(ball_row["ball_x"]))
        ys.append(_as_float(ball_row["ball_y"]))
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _half_court_side_from_mx(mx: float) -> str:
    """Return 'left' (basket x<0 half) vs 'right' from midcourt at x==0."""
    return "left" if mx <= 0 else "right"


def _viewport_x_for_half(side: str) -> Tuple[float, float]:
    """XLim for PNG: geographic halfcourt + small bleed across midcourt and past baseline."""
    mid_bleed = 32.0
    baseline_pad = 18.0
    if side == "left":
        return -564.0 - baseline_pad, 0.0 + mid_bleed
    return 0.0 - mid_bleed, 564.0 + baseline_pad


def make_release_snapshot_figure(
    frame_df: pd.DataFrame,
    shooter_id: int,
    defender_id: int,
    title: str,
    *,
    court_png_view: str = "half",
    court_png_half: Optional[str] = None,
):
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

    ball_ok = all(c in frame_df.columns for c in BALL_COLS)
    if ball_ok:
        bx = _as_float(ball["ball_x"])
        by = _as_float(ball["ball_y"])
        bz = _as_float(ball["ball_z"])
        ax.scatter([bx], [by], [bz], color="#ff8c00", s=90)
        ax.text(bx, by, bz + 6.0, "Ball", color="#ff8c00", fontsize=9)

    view = str(court_png_view).strip().lower()
    if view not in ("half", "full"):
        raise ValueError("court_png_view must be 'half' or 'full'")

    xc_full, _yc, _zc = _court_axis_limits()
    if view == "full":
        _draw_half_court_lines(ax, "full")
        x_lim = (xc_full[0], xc_full[1])
    else:
        mx, _my = _mean_action_xy(shooter, defender, ball, ball_columns_ok=ball_ok)
        raw_half = court_png_half
        if raw_half is None or (isinstance(raw_half, float) and pd.isna(raw_half)):
            hc_side = _half_court_side_from_mx(mx)
        else:
            hs = str(raw_half).strip().lower()
            if hs not in ("left", "right"):
                raise ValueError("court_png_half must be None, 'left', or 'right'")
            hc_side = hs
        _draw_half_court_lines(ax, hc_side)
        x_lim = _viewport_x_for_half(hc_side)

    pts_x: List[float] = []
    pts_y: List[float] = []
    pts_z: List[float] = []
    for row in (shooter, defender):
        for j1, j2 in links:
            for j in (j1, j2):
                cx = _coord_col(frame_df, j, "x")
                cy = _coord_col(frame_df, j, "y")
                cz = _coord_col(frame_df, j, "z")
                if not cx or not cy or not cz:
                    continue
                pts_x.append(_as_float(row[cx]))
                pts_y.append(_as_float(row[cy]))
                pts_z.append(_as_float(row[cz]))
    if ball_ok:
        pts_x.append(_as_float(ball["ball_x"]))
        pts_y.append(_as_float(ball["ball_y"]))
        pts_z.append(_as_float(ball["ball_z"]))
    # Match half-court X span + full sidelines for box aspect math
    pts_x.extend(x_lim)
    pts_y.extend((_yc[0], _yc[1]))
    pts_z.extend((_zc[0], _zc[1]))

    pad_z = 10.0
    z_min_pts = min(pts_z) if pts_z else _zc[0]
    z_max_pts = max(pts_z) if pts_z else _zc[1]
    y_lim: Tuple[float, float] = (_yc[0], _yc[1])
    z_lo = float(min(_zc[0], z_min_pts - pad_z))
    z_hi = float(max(_zc[1], z_max_pts + pad_z))

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim((z_lo, z_hi))
    xr = abs(x_lim[1] - x_lim[0]) or 1.0
    yr = abs(y_lim[1] - y_lim[0]) or 1.0
    zr = abs(z_hi - z_lo) or 1.0
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((xr, yr, zr))

    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=112)
    fig.tight_layout()
    return fig


def release_figure_to_png_bytes(fig, dpi: int = 165) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def save_release_png(fig, out_png: str, dpi: int = 180) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def display_release_png_inline(fig, dpi: int = 165) -> None:
    """Render the matplotlib release snapshot as an inline PNG in Jupyter."""
    try:
        from IPython.display import Image, display
    except ImportError as exc:
        raise ImportError("Inline display requires IPython (install ipython).") from exc
    display(Image(data=release_figure_to_png_bytes(fig, dpi=dpi)))


def prepare_shot_viz_assets(
    shot_row: pd.Series,
    parquet_dir: str,
    pre_frames: int,
    post_frames: int,
    *,
    court_png_view: str = "half",
    court_png_half: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build all visualization objects for one shot row (matplotlib + Plotly).
    Intended for Jupyter notebooks as well as process_one_shot disk export.
    """
    game_file = str(shot_row["game_file"]).strip()
    game_id = (
        _normalize_nba_game_id(shot_row.get("game_id"))
        or _game_id_from_filename(game_file)
        or str(shot_row.get("game_id", "")).strip()
    )
    release_frame = int(float(shot_row["release_frame"]))
    shooter_id = int(float(shot_row["shooter_id"]))
    shooter_name = str(shot_row.get("shooter_name", shooter_id))

    parquet_path = _resolve_parquet_path(parquet_dir, game_file, shot_row.get("game_id"))
    game_df = pd.read_parquet(parquet_path)
    game_df = _select_required_columns(game_df)

    release_df = game_df[game_df["frame"] == release_frame].copy()
    if release_df.empty:
        raise ValueError(f"No rows found at release frame {release_frame} for {game_file}.")

    defender_id = _nearest_defender_id(release_df, shooter_id)
    frame_lo = release_frame - pre_frames
    frame_hi_pre_release = release_frame

    release_only = game_df[game_df["frame"] == release_frame].copy()
    release_only = release_only[release_only["player_id"].isin([shooter_id, defender_id])].copy()

    pre_release_window = game_df[(game_df["frame"] >= frame_lo) & (game_df["frame"] <= frame_hi_pre_release)].copy()
    pre_release_window = pre_release_window[pre_release_window["player_id"].isin([shooter_id, defender_id])].copy()
    if pre_release_window.empty:
        raise ValueError("No rows in pre-release frame window for shooter/defender.")

    pre_release_window = _normalize_joint_columns_for_viz(pre_release_window)
    release_only = _normalize_joint_columns_for_viz(release_only)

    frame_hi_full, full_window_suffix = full_window_last_frame_from_shot_row(
        game_df,
        shot_row,
        release_frame,
        post_frames_fallback=post_frames,
    )

    full_window = game_df[(game_df["frame"] >= frame_lo) & (game_df["frame"] <= frame_hi_full)].copy()
    full_window = full_window[full_window["player_id"].isin([shooter_id, defender_id])].copy()
    full_window = _normalize_joint_columns_for_viz(full_window)

    mpl_title = (
        f"Release snapshot | game {game_id} | frame {release_frame} | "
        f"{shooter_name} vs defender {defender_id}"
    )
    fig_mpl = make_release_snapshot_figure(
        release_df,
        shooter_id,
        defender_id,
        mpl_title,
        court_png_view=court_png_view,
        court_png_half=court_png_half,
    )

    release_title = f"Release only | game {game_id} | frame {release_frame} | shooter {shooter_name}"
    fig_plotly_release = create_plotly_anim(release_only, ball_column="ball", title=release_title)

    pre_anim_title = (
        f"Pre-release ({pre_frames} frames → release) | game {game_id} | {shooter_name}"
    )
    fig_plotly_pre = create_plotly_anim(pre_release_window, ball_column="ball", title=pre_anim_title)

    outcome_frag = pbp_result_title_fragment(shot_row)
    full_title = (
        f"Shot window | {outcome_frag} | ({pre_frames} pre → release frame {release_frame}, {full_window_suffix}) | "
        f"game {game_id} | {shooter_name}"
    )
    fig_plotly_full = create_plotly_anim(full_window, ball_column="ball", title=full_title)

    return {
        "game_id": game_id,
        "release_frame": release_frame,
        "shooter_id": shooter_id,
        "defender_id": defender_id,
        "shooter_name": shooter_name,
        "parquet_path": parquet_path,
        "fig_matplotlib_release": fig_mpl,
        "fig_plotly_release": fig_plotly_release,
        "fig_plotly_pre_release": fig_plotly_pre,
        "fig_plotly_window": fig_plotly_full,
        "release_df": release_df,
        "release_only": release_only,
        "pre_release_window": pre_release_window,
        "full_window": full_window,
    }


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
    *,
    court_png_view: str = "half",
    court_png_half: Optional[str] = None,
) -> Dict[str, str]:
    assets = prepare_shot_viz_assets(
        shot_row,
        parquet_dir,
        pre_frames,
        post_frames,
        court_png_view=court_png_view,
        court_png_half=court_png_half,
    )
    game_id = assets["game_id"]
    release_frame = assets["release_frame"]
    shooter_id = assets["shooter_id"]
    defender_id = assets["defender_id"]
    fig_mpl = assets["fig_matplotlib_release"]
    fig_release = assets["fig_plotly_release"]
    fig_pre = assets["fig_plotly_pre_release"]
    fig_full = assets["fig_plotly_window"]

    stem = f"{game_id}_f{release_frame}_s{shooter_id}_d{defender_id}"
    shot_out_dir = Path(output_dir) / stem
    shot_out_dir.mkdir(parents=True, exist_ok=True)

    png_path = str(shot_out_dir / "release_snapshot_court.png")
    release_html_path = str(shot_out_dir / "release_interactive.html")
    pre_release_html_path = str(shot_out_dir / "pre_release_animation.html")
    full_window_html_path = str(shot_out_dir / "window_animation.html")

    save_release_png(fig_mpl, png_path, dpi=180)
    fig_release.write_html(release_html_path, include_plotlyjs="cdn")
    fig_pre.write_html(pre_release_html_path, include_plotlyjs="cdn")
    fig_full.write_html(full_window_html_path, include_plotlyjs="cdn")

    return {
        "game_id": str(game_id),
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
        default="data/outputs/datasets/shot_contest_dataset.csv",
        help="Shot dataset CSV containing game_file, game_id, release_frame, shooter_id.",
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing *_processed.parquet files referenced by shots CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/visualizations/shot_sequences/runs",
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
    parser.add_argument(
        "--court-png-full",
        action="store_true",
        help="Draw full court for release_snapshot_court.png (default: offensive half only).",
    )
    args = parser.parse_args()

    if not str(args.parquet_dir).strip():
        parser.error("--parquet-dir cannot be empty. Set PARQUET_DIR in the shell or pass --parquet-dir.")

    shots = pd.read_csv(
        args.shots_csv,
        dtype={"game_id": str, "game_file": str},
        keep_default_na=True,
    )
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
                court_png_view="full" if args.court_png_full else "half",
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
