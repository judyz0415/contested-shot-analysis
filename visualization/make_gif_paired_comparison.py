#!/usr/bin/env python3
"""
make_gif_paired_comparison.py

1. Skeleton quality filter on all 6,905 touches
2. Pair matching: CAS vs pass_kickout, matched F2D context, contrasting pose
3. Visual quality checks (frame completeness, court bounds, player proximity)
4. 3D skeleton GIF for each of the top-3 pairs
"""
from __future__ import annotations
import math, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from io import BytesIO
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = (
    "/Users/ruoqianzhu/Library/CloudStorage/OneDrive-SharedLibraries-"
    "MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025"
)
COMBINED_CSV = "data/intermediate/touch_events_combined.csv"
GIF_DIR      = "data/outputs/figures"
os.makedirs(GIF_DIR, exist_ok=True)
N_PAIRS = 3  # change to 3 for full run

# ── Feature sets ──────────────────────────────────────────────────────────────
F2D_CTX = ["n_open_3pt_threats", "min_kickout_defender_gap_ft",
           "release_shot_clock", "game_clock_seconds"]
POSE    = ["def_hand_elev_above_ball_in", "arm_extension_ratio", "trunk_lean_deg",
           "lateral_velocity_ft_s", "min_def_joint_dist_ft"]

# ── Skeleton joints & bones ───────────────────────────────────────────────────
JOINTS = ["neck", "lShoulder", "rShoulder",
          "lElbow", "rElbow", "lWrist", "rWrist",
          "midHip", "lHip", "rHip",
          "lKnee", "rKnee", "lAnkle", "rAnkle"]

BONES = [
    ("neck",      "lShoulder"), ("neck",      "rShoulder"),
    ("lShoulder", "lElbow"),    ("lElbow",    "lWrist"),
    ("rShoulder", "rElbow"),    ("rElbow",    "rWrist"),
    ("neck",      "midHip"),
    ("midHip",    "lHip"),      ("midHip",    "rHip"),
    ("lHip",      "lKnee"),     ("lKnee",     "lAnkle"),
    ("rHip",      "rKnee"),     ("rKnee",     "rAnkle"),
]

JCOLS = [f"{j}_{a}" for j in JOINTS for a in "xyz"]
LOAD_COLS = (["frame", "player_id", "centroid_x", "centroid_y",
               "ball_x", "ball_y", "ball_z"] + JCOLS)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Skeleton quality filter
# ─────────────────────────────────────────────────────────────────────────────
def trunk_lean(row: dict) -> float:
    lsx, lsy, lsz = row["lShoulder_x"], row["lShoulder_y"], row["lShoulder_z"]
    rsx, rsy, rsz = row["rShoulder_x"], row["rShoulder_y"], row["rShoulder_z"]
    lhx, lhy, lhz = row["lHip_x"],      row["lHip_y"],      row["lHip_z"]
    rhx, rhy, rhz = row["rHip_x"],      row["rHip_y"],      row["rHip_z"]
    shom = ((lsx+rsx)/2, (lsy+rsy)/2, (lsz+rsz)/2)
    hipm = ((lhx+rhx)/2, (lhy+rhy)/2, (lhz+rhz)/2)
    horiz = math.hypot(shom[0]-hipm[0], shom[1]-hipm[1])
    return math.degrees(math.atan2(horiz, abs(shom[2]-hipm[2])+1e-9))


def skeleton_ok(drow: dict, ball_z: float) -> bool:
    """Return True if defender skeleton passes all quality checks (units: inches)."""
    required = ["lWrist", "rWrist", "lElbow", "rElbow",
                "lShoulder", "rShoulder", "lHip", "rHip",
                "midHip", "lKnee", "rKnee", "neck"]
    for j in required:
        for a in "xyz":
            v = drow.get(f"{j}_{a}")
            if v is None or (isinstance(v, float) and math.isnan(v)) or v == 0:
                return False

    nz   = drow["neck_z"]
    mhz  = drow["midHip_z"]
    lkz  = drow["lKnee_z"]
    rkz  = drow["rKnee_z"]
    lhz  = drow["lHip_z"]
    lwz  = drow["lWrist_z"]
    rwz  = drow["rWrist_z"]

    height_in = nz - mhz
    if not (24.0 <= height_in <= 54.0):        # 2.0–4.5 ft
        return False
    if trunk_lean(drow) >= 40.0:
        return False
    if max(lwz, rwz) <= lkz:                   # at least one hand above knee
        return False
    if not (18.0 <= mhz <= 60.0):              # hip 1.5–5.0 ft
        return False
    if lkz <= 3.6 or rkz <= 3.6:              # knees > 0.3 ft
        return False
    if lkz > lhz + 6.0:                        # knee not above hip + 0.5 ft
        return False
    if not (48.0 <= ball_z <= 144.0):          # ball 4–12 ft
        return False
    return True


def run_quality_filter(combined: pd.DataFrame) -> pd.DataFrame:
    print("\n── STEP 1: Skeleton quality filter ──────────────────────────────────")
    keep_mask = pd.Series(False, index=combined.index)

    for gf, grp in combined.groupby("game_file"):
        path = os.path.join(INPUT_DIR, gf)
        if not os.path.exists(path):
            print(f"  [skip] {gf} not found")
            continue

        t = (pq.read_table(path, columns=LOAD_COLS)
               .to_pandas()
               .astype({c: float for c in JCOLS + ["ball_x","ball_y","ball_z",
                                                     "centroid_x","centroid_y"]}))

        # index: (frame, player_id) → row dict
        jmap = {}
        bmap = {}  # frame → ball_z
        for r in t.itertuples(index=False):
            key = (int(r.frame), int(r.player_id))
            jmap[key] = {c: getattr(r, c) for c in JCOLS}
            bmap[int(r.frame)] = r.ball_z

        n_pass = 0
        for idx, row in grp.iterrows():
            cf  = int(row.catch_frame)
            did = int(row.nearest_def_id)
            bz  = bmap.get(cf, float("nan"))

            drow = jmap.get((cf, did))
            if drow is None:
                continue
            if skeleton_ok(drow, bz):
                keep_mask[idx] = True
                n_pass += 1

        print(f"  {gf}: {n_pass}/{len(grp)} passed")

    clean = combined[keep_mask].copy()
    print(f"\n  Total passed: {len(clean)} / {len(combined)}")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Pair matching
# ─────────────────────────────────────────────────────────────────────────────
def match_pairs(clean: pd.DataFrame, f2d_thr=0.75, pose_thr=1.5):
    print("\n── STEP 2: Pair matching ─────────────────────────────────────────────")

    feats_all = F2D_CTX + POSE
    for c in feats_all:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=feats_all).copy()

    sc_f2d  = StandardScaler().fit(clean[F2D_CTX])
    sc_pose = StandardScaler().fit(clean[POSE])
    Z_f2d   = sc_f2d.transform(clean[F2D_CTX])
    Z_pose  = sc_pose.transform(clean[POSE])

    cas_idx = clean.index[clean.action == "catch_and_shoot"].tolist()
    pko_idx = clean.index[clean.action == "pass_kickout"].tolist()
    print(f"  CAS pool: {len(cas_idx)}  |  PKO pool: {len(pko_idx)}")

    idx_map   = {i: pos for pos, i in enumerate(clean.index)}
    candidates = []

    for ci in cas_idx:
        ri = idx_map[ci]
        for pi in pko_idx:
            rp       = idx_map[pi]
            f2d_dist = float(np.linalg.norm(Z_f2d[ri] - Z_f2d[rp]))
            if f2d_dist >= f2d_thr:
                continue
            pose_dist = float(np.linalg.norm(Z_pose[ri] - Z_pose[rp]))
            if pose_dist <= pose_thr:
                continue
            score = pose_dist / (f2d_dist + 0.1)
            candidates.append((score, f2d_dist, pose_dist, ci, pi))

    candidates.sort(reverse=True)
    print(f"  Candidate pairs (f2d<{f2d_thr}, pose>{pose_thr}): {len(candidates)}")

    # Select top-3 from different games
    selected, seen_games = [], set()
    for score, f2d_d, pose_d, ci, pi in candidates:
        g_cas = clean.loc[ci, "game_file"]
        g_pko = clean.loc[pi, "game_file"]
        game_key = tuple(sorted([g_cas, g_pko]))
        if game_key in seen_games:
            continue
        selected.append((score, f2d_d, pose_d, ci, pi))
        seen_games.add(game_key)
        if len(selected) == 3:
            break

    print(f"  Selected {len(selected)} pairs from different games")
    return selected, clean, sc_pose, Z_pose, idx_map


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Visual quality checks
# ─────────────────────────────────────────────────────────────────────────────
def visual_check(pair_idx: int, ci: int, pi: int,
                 clean: pd.DataFrame) -> tuple[bool, dict, dict]:
    """
    Returns (ok, cas_data, pko_data) where *_data contains preloaded frames.
    """
    passes = []
    out = {}

    for label, idx in [("CAS", ci), ("PKO", pi)]:
        row  = clean.loc[idx]
        gf   = row.game_file
        cf   = int(row.catch_frame)
        did  = int(row.nearest_def_id)
        hid  = int(row.handler_id)
        path = os.path.join(INPUT_DIR, gf)

        # check handler ≠ defender
        if did == hid:
            print(f"    [{label}] defender == handler, skip")
            passes.append(False); continue

        t = (pq.read_table(path, columns=LOAD_COLS)
               .to_pandas()
               .astype({c: float for c in JCOLS + ["ball_x","ball_y","ball_z",
                                                     "centroid_x","centroid_y"]}))
        t_pid = t.set_index(["frame", "player_id"])

        # frame completeness check on the narrow catch window only
        needed_frames = range(cf - 5, cf + 6)
        complete = True
        for f in needed_frames:
            for pid in [did, hid]:
                if (f, pid) not in t_pid.index:
                    complete = False
                    break

        if not complete:
            print(f"    [{label}] missing frames in window, skip")
            passes.append(False); continue

        # court bounds: centroid of both players at catch_frame
        def_ct = t_pid.loc[(cf, did), ["centroid_x","centroid_y"]].values
        h_ct   = t_pid.loc[(cf, hid), ["centroid_x","centroid_y"]].values
        if not (abs(def_ct[0]) < 570 and abs(def_ct[1]) < 305 and
                abs(h_ct[0])   < 570 and abs(h_ct[1])   < 305):
            print(f"    [{label}] out of court bounds, skip")
            passes.append(False); continue

        # both centroids within 15 ft (180 in) of ball
        ball_row = t[t.frame == cf].iloc[0]
        bx, by   = ball_row.ball_x, ball_row.ball_y
        for ct in [def_ct, h_ct]:
            if math.hypot(ct[0]-bx, ct[1]-by) > 180:
                print(f"    [{label}] player too far from ball, skip")
                passes.append(False); break
        else:
            passes.append(True)

        # collect full frame window for GIF (pre-catch approach + 2 s post-catch)
        full_range = range(cf - 90, cf + 127)
        win = t[t.frame.isin(full_range) &
                t.player_id.isin([did, hid])].copy()

        # per-frame ball position with carry-forward on NaN
        ball_frames = (t[t.frame.isin(full_range)]
                       [["frame","ball_x","ball_y","ball_z"]]
                       .drop_duplicates("frame")
                       .set_index("frame"))
        ball_pos: dict[int, tuple[float, float, float]] = {}
        last_xyz: tuple[float, float, float] = (
            ball_row.ball_x, ball_row.ball_y, ball_row.ball_z)
        for fr in sorted(full_range):
            if fr in ball_frames.index:
                r2 = ball_frames.loc[fr]
                xyz = (float(r2.ball_x), float(r2.ball_y), float(r2.ball_z))
                if not any(math.isnan(v) for v in xyz):
                    last_xyz = xyz
            ball_pos[fr] = last_xyz

        out[label] = {
            "row": row, "did": did, "hid": hid,
            "cf": cf, "window": win,
            "ball_pos": ball_pos,
            # fallback static values (used if frame not in ball_pos)
            "ball_x": ball_row.ball_x,
            "ball_y": ball_row.ball_y,
            "ball_z": ball_row.ball_z,
        }

    return all(passes) and len(passes) == 2, out.get("CAS"), out.get("PKO")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GIF rendering
# ─────────────────────────────────────────────────────────────────────────────
def get_joints(df_window: pd.DataFrame, frame: int, player_id: int) -> dict | None:
    sub = df_window[(df_window.frame == frame) & (df_window.player_id == player_id)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {c: float(getattr(r, c)) for c in JCOLS}


def draw_skeleton(ax, jrow: dict, color: str, alpha: float = 0.9):
    def pt(j):
        return (jrow.get(f"{j}_x"), jrow.get(f"{j}_y"), jrow.get(f"{j}_z"))

    for j1, j2 in BONES:
        p1, p2 = pt(j1), pt(j2)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in p1+p2):
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=2.0, alpha=alpha, solid_capstyle="round")

    # joint dots
    for j in JOINTS:
        p = pt(j)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in p):
            continue
        ax.scatter(*p, color=color, s=18, alpha=alpha, depthshade=False)


def panel_annotation(ax, action: str, pose_vals: dict):
    color = "#C0392B" if "shoot" in action else "#27AE60"
    label = "CATCH & SHOOT" if "shoot" in action else "PASS / KICK-OUT"
    ax.text2D(0.03, 0.96, label, transform=ax.transAxes,
              fontsize=12, fontweight="bold", color=color, va="top")

    # pose features at bottom-left
    lines = [
        f"joint_dist: {pose_vals.get('min_def_joint_dist_ft','?'):.1f} ft",
        f"arm_ext:    {pose_vals.get('arm_extension_ratio','?'):.2f}",
        f"trunk:      {pose_vals.get('trunk_lean_deg','?'):.0f}°",
        f"hand_elev:  {pose_vals.get('def_hand_elev_above_ball_in','?'):+.1f} in",
        f"speed:      {pose_vals.get('lateral_velocity_ft_s','?'):.1f} ft/s",
    ]
    ax.text2D(0.03, 0.08, "\n".join(lines), transform=ax.transAxes,
              fontsize=7.5, color="#2c2c2c", va="bottom",
              family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.65, ec="none"))


def setup_ax(ax, ball_x: float, ball_y: float, ball_z: float, R: float = 96.0):
    ax.set_xlim(ball_x - R, ball_x + R)
    ax.set_ylim(ball_y - R, ball_y + R)
    ax.set_zlim(max(0, ball_z - R), ball_z + R)
    ax.set_xlabel("X (in)", fontsize=7, labelpad=1)
    ax.set_ylabel("Y (in)", fontsize=7, labelpad=1)
    ax.set_zlabel("Z (in)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.4, alpha=0.4)


def render_frame(fig, ax_cas, ax_pko,
                 cas_data: dict, pko_data: dict,
                 cas_frame: int, pko_frame: int,
                 cas_state: dict, pko_state: dict):
    """Render one GIF frame. Axes limits and camera are fixed — only data changes."""
    for ax, data, frame, state in [
        (ax_cas, cas_data, cas_frame, cas_state),
        (ax_pko, pko_data, pko_frame, pko_state),
    ]:
        # Clear only data artists — do NOT reset axis limits or camera
        for line in list(ax.lines):
            line.remove()
        for coll in list(ax.collections):
            coll.remove()
        for txt in list(ax.texts):
            txt.remove()

        # per-frame ball position with carry-forward fallback
        bx, by, bz = data["ball_pos"].get(
            frame, (data["ball_x"], data["ball_y"], data["ball_z"]))

        win = data["window"]

        # skeleton with carry-forward on missing frames
        def_j = get_joints(win, frame, data["did"])
        if def_j is not None:
            state["def_j"] = def_j
        else:
            def_j = state["def_j"]

        h_j = get_joints(win, frame, data["hid"])
        if h_j is not None:
            state["h_j"] = h_j
        else:
            h_j = state["h_j"]

        if def_j:
            draw_skeleton(ax, def_j, "#C0392B")
        if h_j:
            draw_skeleton(ax, h_j,  "#2980B9")

        ax.scatter([bx], [by], [bz], color="#F1C40F", s=80, zorder=5, depthshade=False)

        pose_vals = {feat: data["row"].get(feat) for feat in POSE}
        panel_annotation(ax, data["row"]["action"], pose_vals)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.array(buf, dtype=np.uint8)[..., :3].copy()


def build_offset_seq() -> list[int]:
    """Shared frame offsets (relative to catch_frame) applied to BOTH panels."""
    pre   = list(range(-90, 0, 9))     # ~10 frames: defender approach
    catch = list(range(-5, 6))         # 11 frames: catch zone
    hold  = [0] * 5                    # 5 hold frames frozen at catch
    post  = list(range(6, 127, 6))     # 20 frames: 120 raw = exactly 2 s @ 60fps
    return pre + catch + hold + post   # ~46 total


def make_gif(pair_num: int, cas_data: dict, pko_data: dict):
    print(f"\n── Making GIF {pair_num} ─────────────────────────────────────────────")
    fig = plt.figure(figsize=(16, 7), dpi=100)
    fig.patch.set_facecolor("white")
    ax_cas = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pko = fig.add_subplot(1, 2, 2, projection="3d")

    offsets = build_offset_seq()
    cas_cf  = cas_data["cf"]
    pko_cf  = pko_data["cf"]
    print(f"  Frame count per panel: {len(offsets)}")

    # Fixed axes: center on catch_frame ball position, never change during animation
    for ax, data in [(ax_cas, cas_data), (ax_pko, pko_data)]:
        setup_ax(ax, data["ball_x"], data["ball_y"], data["ball_z"])
        ax.view_init(elev=25, azim=45)

    # carry-forward state for missing skeleton frames
    cas_state: dict = {"def_j": None, "h_j": None}
    pko_state: dict = {"def_j": None, "h_j": None}

    frames = []
    for off in offsets:
        img = render_frame(fig, ax_cas, ax_pko, cas_data, pko_data,
                           cas_cf + off, pko_cf + off,
                           cas_state, pko_state)
        frames.append(img)

    assert len(frames) >= 20, f"GIF too short: {len(frames)} frames"

    plt.close(fig)

    out = os.path.join(GIF_DIR, f"gif_pair_{pair_num}.gif")
    imageio.mimsave(out, frames, duration=100, loop=0)  # 10 fps
    print(f"  Saved → {out}  ({len(frames)} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    combined = pd.read_csv(COMBINED_CSV)
    print(f"Loaded {len(combined)} touches")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    clean = run_quality_filter(combined)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    selected, clean, sc_pose, Z_pose, idx_map = match_pairs(clean)
    if not selected:
        print("No pairs found — try loosening thresholds.")
        return

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Selected pairs ───────────────────────────────────────────────────")
    hdr = f"  {'Pair':>4}  {'CAS game':<38}  {'f2d_dist':>8}  {'pose_dist':>9}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))

    for pn, (score, f2d_d, pose_d, ci, pi) in enumerate(selected, 1):
        g = clean.loc[ci, "game_file"]
        # top-differing pose features
        ri, rp = idx_map[ci], idx_map[pi]
        diffs = {f: abs(Z_pose[ri, k] - Z_pose[rp, k])
                 for k, f in enumerate(POSE)}
        top2 = sorted(diffs, key=diffs.get, reverse=True)[:2]
        print(f"  {pn:>4}  {g:<38}  {f2d_d:>8.3f}  {pose_d:>9.3f}  "
              f"top-diff: {top2[0]} ({diffs[top2[0]]:.2f} SD), "
              f"{top2[1]} ({diffs[top2[1]]:.2f} SD)")

    # ── Steps 3 & 4 ───────────────────────────────────────────────────────────
    gif_count = 0
    for pn, (score, f2d_d, pose_d, ci, pi) in enumerate(selected[:N_PAIRS], 1):
        print(f"\n── Pair {pn}: visual check ───────────────────────────────────────")
        ok, cas_data, pko_data = visual_check(pn, ci, pi, clean)
        if not ok:
            print(f"  Pair {pn} failed visual check, skipping")
            continue
        gif_count += 1
        make_gif(gif_count, cas_data, pko_data)

    print(f"\nDone. {gif_count} GIFs saved to {GIF_DIR}/")


if __name__ == "__main__":
    main()
