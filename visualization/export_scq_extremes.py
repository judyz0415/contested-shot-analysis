"""
export_scq_extremes.py
----------------------
Exports the best and worst SCQ shots as:
  - metric_card_best_scq.png   / metric_card_worst_scq.png
  - pre_release_best_scq.gif   / pre_release_worst_scq.gif

Run from project root:
  .venv/bin/python3 scripts/visualization/export_scq_extremes.py
"""

import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
SHOTS_CSV   = REPO / "data" / "intermediate" / "shot_contest_dataset.csv"
PARQUET_DIR = Path(
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)
OUT_DIR = REPO / "report" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRE_FRAMES  = 60   # ~1 second before release
POST_FRAMES = 6

# ── palette ─────────────────────────────────────────────────────────────────
HEAT_RED  = "#981F2A"
GREEN     = "#27ae60"
RED_BAR   = "#e74c3c"
BG        = "#FAFAFA"

# ── metric config ────────────────────────────────────────────────────────────
METRIC_COLS = [
    "shot_contest_quality",
    "contest_distance_ft",
    "closeout_speed_ft_s",
    "contest_angle_deg",
    "hand_up_in",
    "effective_contest_height_in",
    "defender_jump_in",
    "height_diff_in",
    "wingspan_vs_shooter_height_in",
]
METRIC_LABELS = {
    "shot_contest_quality":          "SCQ (composite)",
    "contest_distance_ft":           "Contest distance (ft)  ← lower = better",
    "closeout_speed_ft_s":           "Closeout speed (ft/s)",
    "contest_angle_deg":             "Contest angle (°)",
    "hand_up_in":                    "Hand above centroid (in)",
    "effective_contest_height_in":   "Hand vs ball height (in)",
    "defender_jump_in":              "Defender rise 250ms pre-release (in)",
    "height_diff_in":                "Def − shooter height (in)",
    "wingspan_vs_shooter_height_in": "Wingspan − shooter height (in)",
}
DIRECTION = {                        # +1 = higher is better defense
    "shot_contest_quality":          +1,
    "contest_distance_ft":           -1,
    "closeout_speed_ft_s":           +1,
    "contest_angle_deg":             +1,
    "hand_up_in":                    +1,
    "effective_contest_height_in":   +1,
    "defender_jump_in":              +1,
    "height_diff_in":                +1,
    "wingspan_vs_shooter_height_in": +1,
}


# ── load data ────────────────────────────────────────────────────────────────
shots = pd.read_csv(SHOTS_CSV, dtype={"game_id": str})
shots = shots[shots["analysis_eligible"] == "yes"].reset_index(drop=True)
for c in METRIC_COLS:
    shots[c] = pd.to_numeric(shots[c], errors="coerce")

pool_means = shots[METRIC_COLS].mean()

best_row  = shots.loc[shots["shot_contest_quality"].idxmax()]
worst_row = shots.loc[shots["shot_contest_quality"].idxmin()]

print(f"BEST  SCQ {best_row['shot_contest_quality']:.1f}  "
      f"{best_row['shooter_name']} vs {best_row['nearest_defender_name']}  "
      f"result={best_row['pbp_shot_result']}")
print(f"WORST SCQ {worst_row['shot_contest_quality']:.1f}  "
      f"{worst_row['shooter_name']} vs {worst_row['nearest_defender_name']}  "
      f"result={worst_row['pbp_shot_result']}")


# ── metric card ──────────────────────────────────────────────────────────────
def save_metric_card(row: pd.Series, out_path: Path, tag: str):
    labels, deltas, colors = [], [], []
    for col in METRIC_COLS:
        val = row.get(col)
        if pd.isna(val):
            continue
        delta = float(val) - float(pool_means[col])
        good  = delta * DIRECTION.get(col, +1) > 0
        labels.append(METRIC_LABELS[col])
        deltas.append(delta)
        colors.append(GREEN if good else RED_BAR)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = range(len(labels))
    bars = ax.barh(list(y_pos), deltas, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")

    for bar, delta in zip(bars, deltas):
        x = bar.get_width()
        ax.text(x + (0.3 if x >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f"{delta:+.1f}", va="center", ha="left" if x >= 0 else "right",
                fontsize=8.5, color="#333333")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ from pool mean  (green = better for defense)", fontsize=9)

    result  = row.get("pbp_shot_result", "?")
    shooter = row.get("shooter_name", "?")
    defender = row.get("nearest_defender_name", "?")
    scq = float(row.get("shot_contest_quality", 0))
    ax.set_title(
        f"[{tag}]  {shooter} vs {defender}  ·  {result}  ·  SCQ = {scq:.1f}",
        fontsize=11, fontweight="bold", color=HEAT_RED, pad=10
    )

    green_patch = mpatches.Patch(color=GREEN,   label="Better than pool avg")
    red_patch   = mpatches.Patch(color=RED_BAR, label="Worse than pool avg")
    ax.legend(handles=[green_patch, red_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved metric card → {out_path}")


save_metric_card(best_row,  OUT_DIR / "metric_card_best_scq.png",  "Best SCQ")
save_metric_card(worst_row, OUT_DIR / "metric_card_worst_scq.png", "Worst SCQ")


# ── GIF animation ────────────────────────────────────────────────────────────
def load_parquet(game_id: str) -> pd.DataFrame:
    pattern = PARQUET_DIR / f"nba_game_{game_id}_processed.parquet"
    if not pattern.exists():
        raise FileNotFoundError(f"Parquet not found: {pattern}")
    return pd.read_parquet(pattern)


HEAT_ID = 1610612748  # Miami Heat franchise ID

def save_shot_gif(row: pd.Series, out_path: Path, tag: str):
    game_id      = str(row["game_id"])
    rel_frame    = int(row["release_frame"])
    shooter_id   = int(row["shooter_id"])
    defender_id  = int(row.get("nearest_defender_id", -1))

    print(f"Loading parquet for game {game_id} …")
    df = load_parquet(game_id)

    start = max(0, rel_frame - PRE_FRAMES)
    end   = rel_frame + POST_FRAMES
    clip  = df[(df["frame"] >= start) & (df["frame"] <= end)].copy()

    # Identify unique players
    player_ids = clip["player_id"].dropna().unique().astype(int)

    # Determine team colours
    heat_ids = set()
    if "team_id" in clip.columns:
        heat_ids = set(
            clip.loc[clip["team_id"] == HEAT_ID, "player_id"].dropna().astype(int).unique()
        )

    # ── figure setup ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333355")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")

    # fixed axis limits from clip data
    cx = clip["centroid_x"].dropna()
    cy = clip["centroid_y"].dropna()
    cz = clip["centroid_z"].dropna()
    bx = clip["ball_x"].dropna() if "ball_x" in clip.columns else cx
    by = clip["ball_y"].dropna() if "ball_y" in clip.columns else cy
    bz = clip["ball_z"].dropna() if "ball_z" in clip.columns else cz

    pad = 2.5
    xlim = (min(cx.min(), bx.min()) - pad, max(cx.max(), bx.max()) + pad)
    ylim = (min(cy.min(), by.min()) - pad, max(cy.max(), by.max()) + pad)
    zlim = (0, max(cz.max(), bz.max()) + pad)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("x (ft)", fontsize=7, color="white")
    ax.set_ylabel("y (ft)", fontsize=7, color="white")
    ax.set_zlabel("z (ft)", fontsize=7, color="white")

    # rim marker (approximate)
    rim_x = float(row.get("rim_x", xlim[0] + (xlim[1]-xlim[0])*0.5))
    rim_y = float(row.get("rim_y", ylim[0] + (ylim[1]-ylim[0])*0.5))
    ax.scatter([rim_x], [rim_y], [7.5], c="orange", s=60, marker="o", zorder=5, label="Rim")

    frames_sorted = sorted(clip["frame"].unique())

    def _get_latest(sub_clip, frame_idx, col):
        """Return the most recent non-null value of col up to frame_idx."""
        past = sub_clip.loc[sub_clip["frame"] <= frame_idx, col].dropna()
        return float(past.iloc[-1]) if not past.empty else np.nan

    # trail lengths
    TRAIL = 20

    def animate(frame_idx):
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel("x (ft)", fontsize=7, color="white")
        ax.set_ylabel("y (ft)", fontsize=7, color="white")
        ax.set_zlabel("z (ft)", fontsize=7, color="white")
        ax.tick_params(colors="#aaaaaa", labelsize=5)

        trail_start = max(frames_sorted[0], frame_idx - TRAIL)
        trail_frames = [f for f in frames_sorted if trail_start <= f <= frame_idx]

        sub = clip[clip["frame"].isin(trail_frames)]

        # Ball trail
        if "ball_x" in clip.columns:
            bsub = sub.drop_duplicates("frame").sort_values("frame")
            bxs = bsub["ball_x"].dropna().values
            bys = bsub["ball_y"].dropna().values
            bzs = bsub["ball_z"].dropna().values
            if len(bxs) >= 2:
                ax.plot(bxs, bys, bzs, color="#f39c12", linewidth=1.2, alpha=0.8)
            # current ball
            if len(bxs):
                ax.scatter([bxs[-1]], [bys[-1]], [bzs[-1]],
                           c="#f39c12", s=55, zorder=10, label="Ball")

        # Players
        for pid in player_ids:
            psub = sub[sub["player_id"] == pid].drop_duplicates("frame").sort_values("frame")
            if psub.empty:
                continue
            xs = psub["centroid_x"].dropna().values
            ys = psub["centroid_y"].dropna().values
            zs = psub["centroid_z"].dropna().values
            if len(xs) == 0:
                continue

            is_heat     = pid in heat_ids
            is_shooter  = pid == shooter_id
            is_defender = pid == defender_id

            if is_shooter:
                colour, marker, size, alpha = "#e74c3c", "^", 80, 1.0
            elif is_defender:
                colour, marker, size, alpha = "#3498db", "s", 80, 1.0
            elif is_heat:
                colour, marker, size, alpha = "#981F2A", "o", 45, 0.7
            else:
                colour, marker, size, alpha = "#95a5a6", "o", 35, 0.5

            # trail
            if len(xs) >= 2:
                ax.plot(xs, ys, zs, color=colour, linewidth=0.8, alpha=0.4)
            # current dot
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                       c=colour, marker=marker, s=size, alpha=alpha, zorder=8)

        # Rim
        ax.scatter([rim_x], [rim_y], [7.5], c="orange", s=55, marker="o", zorder=5)

        # Release line indicator
        rel_ms = frame_idx == rel_frame
        frame_label = f"frame {frame_idx}"
        if frame_idx == rel_frame:
            frame_label += "  ← RELEASE"
        ax.set_title(
            f"[{tag}]  {row.get('shooter_name')} vs {row.get('nearest_defender_name')}\n"
            f"SCQ={float(row.get('shot_contest_quality', 0)):.1f}  "
            f"result={row.get('pbp_shot_result')}  |  {frame_label}",
            color="white", fontsize=8, pad=4
        )

        legend_patches = [
            mpatches.Patch(color="#e74c3c", label="Shooter"),
            mpatches.Patch(color="#3498db", label="Defender"),
            mpatches.Patch(color="#981F2A", label="Other Heat"),
            mpatches.Patch(color="#95a5a6", label="Opponent"),
            mpatches.Patch(color="#f39c12", label="Ball"),
        ]
        ax.legend(handles=legend_patches, fontsize=6, loc="upper left",
                  facecolor="#111122", labelcolor="white", framealpha=0.7)

    ani = FuncAnimation(fig, animate, frames=frames_sorted, interval=80, blit=False)
    writer = PillowWriter(fps=12)
    ani.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"Saved GIF → {out_path}")


save_shot_gif(best_row,  OUT_DIR / "pre_release_best_scq.gif",  "Best SCQ")
save_shot_gif(worst_row, OUT_DIR / "pre_release_worst_scq.gif", "Worst SCQ")

print("\nAll assets saved to:", OUT_DIR)
