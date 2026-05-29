"""
export_notebook_gifs.py
-----------------------
Uses the exact same prepare_shot_viz_assets call as metric_alignment_viz.ipynb,
then renders each Plotly animation frame with kaleido and stitches into a GIF.

Run from project root:
  .venv/bin/python3 scripts/visualization/export_notebook_gifs.py
"""

import sys, copy, io
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "visualization"))

# ── same module load as notebook cell 1 ─────────────────────────────────────
import importlib.util
for m in ("shot_viz_from_dataset", "viz"):
    sys.modules.pop(m, None)
_spec = importlib.util.spec_from_file_location(
    "shot_viz_from_dataset",
    REPO / "scripts" / "visualization" / "shot_viz_from_dataset.py",
)
_sv = importlib.util.module_from_spec(_spec)
sys.modules["shot_viz_from_dataset"] = _sv
_spec.loader.exec_module(_sv)
prepare_shot_viz_assets = _sv.prepare_shot_viz_assets

# ── same constants as notebook cell 2 ────────────────────────────────────────
SHOTS_CSV   = REPO / "data" / "intermediate" / "shot_contest_dataset.csv"
PARQUET_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)
PRE_FRAMES  = 60
POST_FRAMES = 4
PARQUET_GAME_CACHE: dict = {}

OUT_DIR = REPO / "report" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 12

# ── load shots (same as notebook cell 3) ─────────────────────────────────────
shots = pd.read_csv(SHOTS_CSV, dtype={"game_id": str, "game_file": str})
shots = shots[shots["analysis_eligible"] == "yes"].reset_index(drop=True)
shots["shot_contest_quality"] = pd.to_numeric(shots["shot_contest_quality"], errors="coerce")

best_row  = shots.loc[shots["shot_contest_quality"].idxmax()]
worst_row = shots.loc[shots["shot_contest_quality"].idxmin()]

print(f"BEST  SCQ={best_row['shot_contest_quality']:.1f}  "
      f"{best_row['shooter_name']} vs {best_row['nearest_defender_name']}  "
      f"result={best_row['pbp_shot_result']}")
print(f"WORST SCQ={worst_row['shot_contest_quality']:.1f}  "
      f"{worst_row['shooter_name']} vs {worst_row['nearest_defender_name']}  "
      f"result={worst_row['pbp_shot_result']}")


# ── render Plotly animation → GIF ────────────────────────────────────────────
def plotly_anim_to_gif(fig: go.Figure, out_path: Path, fps: int = FPS):
    """
    Render each animation frame of a Plotly figure to PNG via kaleido,
    stitch into a GIF with PIL.

    Frame structure from create_plotly_anim in viz.py:
      frame.data   — list of trace dicts {type, x, y, z} for dynamic traces
      frame.traces — list of int indices into fig.data to update
      frame.layout — layout with annotations (frame number, TimeUTC, SCQ)
    Static court traces are NOT in frame.traces and stay unchanged each frame.
    """
    if not fig.frames:
        print(f"  No frames — skipping {out_path.name}")
        return

    n = len(fig.frames)
    print(f"  Rendering {n} frames …")
    pil_frames = []

    for i, frame in enumerate(fig.frames):
        # Build a static snapshot for this frame:
        # 1. Deep-copy all base traces
        traces = [copy.deepcopy(t) for t in fig.data]

        # 2. Apply per-frame updates to the dynamic traces
        for trace_idx, trace_update in zip(frame.traces, frame.data):
            # trace_update is a dict-like object; extract x/y/z
            upd = trace_update if isinstance(trace_update, dict) else trace_update.to_plotly_json()
            for key in ("x", "y", "z"):
                if key in upd:
                    traces[trace_idx][key] = upd[key]

        # 3. Build a static figure with the frame's layout (annotations etc.)
        layout = copy.deepcopy(fig.layout)
        if frame.layout:
            frame_layout = (
                frame.layout if isinstance(frame.layout, dict)
                else frame.layout.to_plotly_json()
            )
            layout.update(frame_layout)

        # Remove animation controls from static layout
        layout.updatemenus = []
        layout.sliders     = []

        snap = go.Figure(data=traces, layout=layout)

        png_bytes = pio.to_image(snap, format="png", width=900, height=620, scale=1)
        pil_frames.append(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n}")

    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"  Saved → {out_path}  ({n} frames, {fps} fps, {size_kb} KB)")


def focus_on_half_court(fig: go.Figure, rim_x: float) -> go.Figure:
    """
    Restrict the scene to the half-court containing the rim and set a camera
    angle that keeps the rim, shooter, and defender all visible.
    rim_x ≈ +516 → right half (x: 50 to 600)
    rim_x ≈ −516 → left half  (x: -600 to -50)
    """
    fig = copy.deepcopy(fig)

    if rim_x > 0:
        x_range = [200, 620]   # tighter: only the scoring half
        # Camera pulled closer and angled toward the key/rim area
        eye    = dict(x=-1.2, y=1.1, z=0.75)
        center = dict(x=0.25, y=0.0, z=-0.05)
    else:
        x_range = [-620, -200]
        eye    = dict(x=1.2,  y=1.1, z=0.75)
        center = dict(x=-0.25, y=0.0, z=-0.05)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, title="X"),
            yaxis=dict(range=[-230, 230], title="Y"),  # tighter sideline crop
            zaxis=dict(range=[0, 220],    title="Z"),
            camera=dict(eye=eye, center=center, up=dict(x=0, y=0, z=1)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.1, z=0.45),
        )
    )
    return fig


def export_gif(row: pd.Series, out_path: Path, label: str):
    print(f"\n[{label}]  loading assets …")
    # exact same call as notebook show_shot()
    assets = prepare_shot_viz_assets(
        row,
        parquet_dir=str(PARQUET_DIR),
        pre_frames=PRE_FRAMES,
        post_frames=POST_FRAMES,
        court_png_view="half",
        court_png_half=None,
        parquet_game_df_cache=PARQUET_GAME_CACHE,
    )
    fig = assets["fig_plotly_pre_release"]   # 61 frames; fig_plotly_window has 961

    # Zoom into the relevant half-court so rim is always visible
    rim_x = float(row.get("rim_x", 516))
    fig = focus_on_half_court(fig, rim_x)

    plotly_anim_to_gif(fig, out_path)


export_gif(best_row,  OUT_DIR / "pre_release_best_scq.gif",  "Best SCQ")
export_gif(worst_row, OUT_DIR / "pre_release_worst_scq.gif", "Worst SCQ")

print("\nDone.")
