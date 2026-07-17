#!/usr/bin/env python3
"""
Re-run the openness->make and pitch-control->make validations on GROUND-TRUTH
labels (pbp), now that all 15 games are parsed.

Opponent 3PA already carry pbp_shot_result. Heat 3PA were tracking-labeled
(97% on opponents, unvalidated on Heat); here we attach the game clock to each
Heat shot from the parquet and join to the parsed pbp FG3 events on
(shooter, period, nearest clock) to get the true make/miss -- and measure the
tracking detector's accuracy on Heat shots as a bonus.
"""
from __future__ import annotations
import math, os, re, sys, warnings
warnings.filterwarnings("ignore")
LOCAL_SITE = os.path.join(os.getcwd(), ".python_packages")
if os.path.isdir(LOCAL_SITE):
    sys.path.insert(0, LOCAL_SITE)
import pandas as pd, numpy as np
import pyarrow.parquet as pq
import statsmodels.api as sm, statsmodels.formula.api as smf
from collections import defaultdict

INP = ("/Users/ruoqianzhu/Library/CloudStorage/OneDrive-SharedLibraries-"
       "MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025")

# pitch-control constants (literature defaults)
MAX_SPEED, REACTION, SIGMA, LAM, DT, TMAX = 16.4, 0.3, 0.45, 4.3, 0.04, 6.0


def clk(s):
    s = str(s).strip()
    if ":" in s:
        return int(s.split(":")[0]) * 60 + int(float(s.split(":")[1]))
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def attach_gameclock(df):
    """Map (game_file, release_frame) -> game-clock seconds via the parquet."""
    out = {}
    for gf, g in df.groupby("game_file"):
        t = pq.read_table(os.path.join(INP, gf), columns=["frame", "period", "gameClockTime"]).to_pandas().drop_duplicates("frame")
        fmap = {int(r.frame): clk(r.gameClockTime) for r in t.itertuples(index=False)}
        for rf in g.release_frame.astype(int):
            out[(gf, rf)] = fmap.get(rf)
    return out


def heat_ground_truth():
    heat = pd.read_csv("data/intermediate/heat_3pa_dataset.csv")
    heat["gid"] = heat.game_file.str.extract(r"(00\d{8})").astype(int)
    gc = attach_gameclock(heat)
    heat["gclock"] = [gc.get((r.game_file, int(r.release_frame))) for r in heat.itertuples(index=False)]
    pbp = pd.read_csv("data/intermediate/pbp_events.csv")
    fg3 = pbp[pbp.action == "FG3"].copy()
    gt = []
    for r in heat.itertuples(index=False):
        if r.gclock is None:
            gt.append(np.nan); continue
        c = fg3[(fg3.game_id == r.gid) & (fg3.person_id == r.shooter_id) & (fg3.period == r.period)]
        if len(c):
            c = c.assign(off=(c.clock_sec - r.gclock).abs()).sort_values("off")
            gt.append(int(c.iloc[0].made) if c.iloc[0].off <= 8 else np.nan)
        else:
            gt.append(np.nan)
    heat["gt_made"] = gt
    return heat


def tti(loc, p):
    rx, ry = p[0] + p[2] * REACTION, p[1] + p[3] * REACTION
    return REACTION + math.hypot(loc[0] - rx, loc[1] - ry) / MAX_SPEED


def def_control(loc, players):
    taus = [tti(loc, p) for p in players]; isd = [p[4] for p in players]
    ppcf = [0.0] * len(players); coef = math.pi / (math.sqrt(3) * SIGMA)
    t, tot = max(0.0, min(taus) - 3 * SIGMA), 0.0
    while tot < 0.99 and t < TMAX:
        for i in range(len(players)):
            if ppcf[i] >= 1: continue
            d = (1 - tot) * (1 / (1 + math.exp(-coef * (t - taus[i])))) * LAM * DT
            ppcf[i] += d; tot += d
        t += DT
    return sum(ppcf[i] for i in range(len(players)) if isd[i])


def score_states(states_csv, label_map):
    shots = defaultdict(list)
    import csv
    with open(states_csv, newline="") as f:
        for r in csv.DictReader(f):
            shots[(r["game_file"], int(r["release_frame"]), int(r["shooter_id"]), int(r["nearest_defender_id"]))].append(r)
    rows = []
    for key, pl in shots.items():
        if key not in label_map or pd.isna(label_map[key]): continue
        sh = next((p for p in pl if p["is_shooter"] == "1"), None)
        if not sh: continue
        side = sh["is_offense"]
        loc = (float(sh["x_ft"]), float(sh["y_ft"]))
        players = [(float(p["x_ft"]), float(p["y_ft"]), float(p["vx_ft_s"]), float(p["vy_ft_s"]), p["is_offense"] != side) for p in pl]
        rows.append({"def_control": def_control(loc, players), "made": int(label_map[key])})
    return pd.DataFrame(rows)


def main():
    print("=== A. Heat ground-truth labels (join to pbp) ===")
    heat = heat_ground_truth()
    lab = heat.dropna(subset=["gt_made"])
    acc = (lab.tracking_made == lab.gt_made).mean()
    print(f"Heat 3PA: {len(heat)} | matched to pbp: {len(lab)} | tracking-detector accuracy on Heat: {acc:.3f}")
    print(f"  pbp Heat make rate {lab.gt_made.mean():.3f} vs tracking {lab.tracking_made.mean():.3f}")
    heat.to_csv("data/intermediate/heat_3pa_pbp_labeled.csv", index=False)

    # opponent shots already pbp-labeled
    opp = pd.read_csv("data/intermediate/shot_contest_dataset.csv")
    opp = opp[opp.pbp_shot_result.astype(str).str.contains("made|miss", case=False)].copy()
    opp["made"] = opp.pbp_shot_result.str.contains("made", case=False).astype(int)

    print("\n=== B. Openness -> make (ground truth) ===")
    comb = pd.concat([opp[["contest_distance_ft", "made"]],
                      lab.rename(columns={"gt_made": "made"})[["contest_distance_ft", "made"]]], ignore_index=True).dropna()
    m = smf.glm("made ~ contest_distance_ft", comb, family=sm.families.Binomial()).fit()
    print(f"  n={len(comb)}  coef(contest_distance)={m.params['contest_distance_ft']:+.4f}  p={m.pvalues['contest_distance_ft']:.4f}")
    print("  (tracking-labeled was coef +0.083, p=0.0010)")

    print("\n=== C. Pitch control -> make (ground truth) ===")
    opp_lab = {(r.game_file, int(r.release_frame), int(r.shooter_id), int(r.nearest_defender_id)): int(r.made) for r in opp.itertuples(index=False)}
    heat_lab = {(r.game_file, int(r.release_frame), int(r.shooter_id), int(r.nearest_defender_id)): r.gt_made for r in lab.itertuples(index=False)}
    so = score_states("data/intermediate/player_states_at_release.csv", opp_lab)
    sh = score_states("data/intermediate/player_states_heat.csv", heat_lab)
    both = pd.concat([so, sh], ignore_index=True)
    m2 = smf.glm("made ~ def_control", both, family=sm.families.Binomial()).fit()
    print(f"  n={len(both)} (opp {len(so)}, heat {len(sh)})  coef(def_control)={m2.params['def_control']:+.3f}  p={m2.pvalues['def_control']:.4f}  OR={np.exp(m2.params['def_control']):.3f}")
    print("  (tracking-labeled was coef -2.42, p=0.0011)")
    both["q"] = pd.qcut(both.def_control, 5, labels=["Q1 least", "Q2", "Q3", "Q4", "Q5 most"])
    print(both.groupby("q", observed=True).agg(n=("made", "size"), make=("made", "mean")).assign(make=lambda d: (d.make * 100).round(1)).to_string())


if __name__ == "__main__":
    main()
