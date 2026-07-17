#!/usr/bin/env python3
"""
Detect what a perimeter ball-handler did under a closeout -- shot vs. a "worse
action" (pass-out, drive, turnover) -- to value closeouts beyond shots-taken.

We only kept shots before; but a great closeout often prevents the shot entirely.
Using last_touch_player_id to segment each offensive player's ball-touches, we
classify every perimeter (>22 ft) catch by how it ended, and record the nearest
defender's distance at the catch. This lets us ask: does a tighter closeout
push the handler off the shot into a lower-value action?

Endings: SHOT (a release from our 3PA list lands in the touch window) /
DRIVE (handler got within 14 ft of the rim) / TURNOVER (ball next touched by a
defender) / PASS (ball next touched by a teammate) / OTHER.

Output: data/intermediate/closeout_actions.csv  (offense = opponent; defenders = Heat)
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from collections import defaultdict
from typing import Dict, List

LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = (
    "/Users/ruoqianzhu/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)
RIMS = [(516.0, 0.0), (-516.0, 0.0)]
PERIM_FT = 22.0
DRIVE_FT = 14.0


def rimdist(x, y):
    return min(math.hypot(x - rx, y - ry) for rx, ry in RIMS)


def process_game(path, shot_frames, offense="opponent"):
    """
    offense="opponent": Heat defending, opponent handlers  (original direction)
    offense="heat":     opponents defending, Heat handlers (flipped direction)
    """
    t = pq.read_table(path, columns=["frame", "player_id", "team_id", "teamName",
                                     "centroid_x", "centroid_y", "last_touch_player_id"]).to_pandas()
    heat_id  = t[t.teamName == "Heat"].team_id.iloc[0]
    pteam    = t.groupby("player_id").team_id.first()
    heat_pids = [int(p) for p in pteam.index if pteam[p] == heat_id]
    opp_pids  = [int(p) for p in pteam.index if pteam[p] != heat_id]

    # who handles, who defends, and which team winning the ball = turnover
    if offense == "heat":
        is_handler  = lambda pid: pteam[pid] == heat_id
        def_pids    = opp_pids
        is_turnover = lambda nxt_team: nxt_team is not None and nxt_team != heat_id
    else:
        is_handler  = lambda pid: pteam[pid] != heat_id
        def_pids    = heat_pids
        is_turnover = lambda nxt_team: nxt_team == heat_id

    pos = {(int(r.frame), int(r.player_id)): (r.centroid_x, r.centroid_y)
           for r in t.itertuples(index=False)}
    fr = t[["frame", "last_touch_player_id"]].drop_duplicates("frame").sort_values("frame").reset_index(drop=True)
    fr["touch"] = fr["last_touch_player_id"]
    fr["grp"]   = (fr["touch"] != fr["touch"].shift()).cumsum()
    segs = list(fr.groupby("grp"))
    out = []
    for i, (_, seg) in enumerate(segs):
        pid = seg["touch"].iloc[0]
        if pd.isna(pid):
            continue
        pid = int(pid)
        if pid not in pteam.index or not is_handler(pid):
            continue
        f0, f1 = int(seg.frame.iloc[0]), int(seg.frame.iloc[-1])
        p0 = pos.get((f0, pid))
        if p0 is None:
            continue
        d0 = rimdist(*p0) / 12.0
        if d0 < PERIM_FT:
            continue
        dmin = min((rimdist(*pos[(f, pid)]) / 12.0 for f in seg.frame if (f, pid) in pos), default=d0)
        def nearest_def(frame):
            best, bid = 99.0, -1
            hp = pos.get((frame, pid))
            if hp is None:
                return best, bid
            for dp in def_pids:
                q = pos.get((frame, dp))
                if q is None:
                    continue
                dd = math.hypot(q[0] - hp[0], q[1] - hp[1]) / 12.0
                if dd < best:
                    best, bid = dd, dp
            return best, bid
        def_dist, def_id = nearest_def(f0)
        prior, _         = nearest_def(f0 - 30)
        closeout_speed   = (prior - def_dist) / 0.5 if prior < 90 else float("nan")
        shot    = any((f in shot_frames) for f in range(f0, f1 + 16))
        nxt     = segs[i + 1][1]["touch"].iloc[0] if i + 1 < len(segs) else None
        nxt_team = pteam.get(int(nxt)) if pd.notna(nxt) else None
        if shot:
            end = "SHOT"
        elif dmin < DRIVE_FT:
            end = "DRIVE"
        elif is_turnover(nxt_team):
            end = "TURNOVER"
        elif pd.notna(nxt) and int(nxt) != pid:
            end = "PASS"
        else:
            end = "OTHER"
        out.append({"game_file": os.path.basename(path), "catch_frame": f0,
                    "handler_id": pid, "nearest_def_id": def_id,
                    "def_dist_ft": round(def_dist, 2),
                    "closeout_speed_ft_s": round(closeout_speed, 2) if math.isfinite(closeout_speed) else "",
                    "catch_rimdist_ft": round(d0, 1), "ending": end})
    del pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--shots-csv", default="data/intermediate/shot_contest_dataset.csv")
    ap.add_argument("--output-csv", default="data/intermediate/closeout_actions.csv")
    ap.add_argument("--offense", default="opponent", choices=["opponent", "heat"],
                    help="'opponent': Heat defends (default); 'heat': opponents defend Heat")
    args = ap.parse_args()

    # heat-offense mode: shots are Heat 3PA releases
    if args.offense == "heat" and args.shots_csv == "data/intermediate/shot_contest_dataset.csv":
        args.shots_csv  = "data/intermediate/heat_3pa_pbp_labeled.csv"
        args.output_csv = "data/intermediate/closeout_actions_heat_offense.csv"

    sd = pd.read_csv(args.shots_csv)
    shots_by_game = {gf: set(g.release_frame.astype(int)) for gf, g in sd.groupby("game_file")}

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    rows = []
    for f in files:
        sf = shots_by_game.get(f, set())
        ev = process_game(os.path.join(args.input_dir, f), sf, offense=args.offense)
        rows.extend(ev)
        from collections import Counter
        print(f"  {f}: {len(ev)} perimeter touches  {dict(Counter(e['ending'] for e in ev))}")
    cols = ["game_file", "catch_frame", "handler_id", "nearest_def_id", "def_dist_ft",
            "closeout_speed_ft_s", "catch_rimdist_ft", "ending"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} perimeter touches to {args.output_csv}")


if __name__ == "__main__":
    main()
