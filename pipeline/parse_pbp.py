#!/usr/bin/env python3
"""
Parse NBA play-by-play (saved from the rendered nba.com page) into a clean
event table -- ground-truth outcomes to join onto the Hawk-Eye tracking.

Input: a folder of per-game files (.rtf / .txt / .json). Each wraps a JSON object
with a "content" markdown field (the full play-by-play page). RTF is converted
with macOS `textutil`. Each play line embeds the personId (headshot URL), the
teamId (logo URL), the description (with MISS / "(N PTS)"), and a stats URL with
the GameID; the preceding line is the game clock. Periods come from "Qn start".

Validated on game 0022500296: 22/22 of our labeled opponent threes matched on
(shooter, period, nearest clock); 95% make/miss agreement; clock offset 0-4 s.

Output: data/intermediate/pbp_events.csv
  game_id, period, clock_sec, person_id, team_id, action, made, points, description
  action in {FG3, FG2, FT, REB, TOV, FOUL, OTHER}; made in {1,0,''}; clock_sec = sec
  remaining in the period.
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, tempfile
from typing import Dict, List, Optional

SHOT2_RE = re.compile(r"Layup|Dunk|Jump Shot|Hook|Floating|Bank|Fadeaway|Finger Roll|Tip|Putback|Reverse")


def load_content(path: str) -> str:
    """Return the page-markdown 'content' string from an rtf/txt/json file."""
    if path.lower().endswith(".rtf"):
        out = subprocess.run(["textutil", "-convert", "txt", "-stdout", path],
                             capture_output=True, text=True)
        raw = out.stdout
    else:
        raw = open(path, encoding="utf-8", errors="ignore").read()
    raw = raw.strip()
    # The file is a JSON object {title, content, ...}; fall back to raw text.
    try:
        obj = json.loads(raw)
        return obj.get("content", raw), obj.get("url", "")
    except json.JSONDecodeError:
        return raw, ""


def game_id_from(path: str, url: str) -> Optional[str]:
    for s in (url, os.path.basename(path)):
        m = re.search(r"(00\d{8})", s)
        if m:
            return m.group(1)
    return None


def parse_content(content: str) -> List[dict]:
    events: List[dict] = []
    period = 0
    cur: Optional[int] = None
    for line in content.split("\n"):
        s = line.strip()
        mq = re.match(r"(?:#+\s*)?Q(\d)\s+start", s)
        if mq:
            period = int(mq.group(1)); continue
        if re.match(r"(?:#+\s*)?OT\d?\s+start", s) or s.strip() in ("## OT start", "OT start"):
            period += 1; continue
        tm = re.match(r"(\d{1,2}):(\d{2})", s)
        if tm and not s.startswith("[!"):
            cur = int(tm.group(1)) * 60 + int(tm.group(2)); continue
        if s.startswith("[!") and "stats/events" in s:
            pid = re.search(r"260x190/(\d+)\.png", s)
            tid = re.search(r"logos/nba/(\d+)/", s)
            dm = re.search(r"logo\.svg\)\s*(.*?)\]\(https://www\.nba\.com/stats/events", s)
            desc = dm.group(1).strip() if dm else ""
            if not desc:
                continue
            made = 0 if desc.startswith("MISS") else (1 if re.search(r"\(\d+ PTS\)", desc) else None)
            if "3PT" in desc:
                act = "FG3"
            elif "Free Throw" in desc:
                act = "FT"
            elif SHOT2_RE.search(desc):
                act = "FG2"
            elif "REBOUND" in desc:
                act = "REB"
            elif "Turnover" in desc:
                act = "TOV"
            elif "FOUL" in desc:
                act = "FOUL"
            else:
                act = "OTHER"
            pts = 3 if (act == "FG3" and made == 1) else 2 if (act == "FG2" and made == 1) else 1 if (act == "FT" and made == 1) else 0
            events.append(dict(period=period, clock_sec=cur,
                               person_id=int(pid.group(1)) if pid else "",
                               team_id=int(tid.group(1)) if tid else "",
                               action=act, made=("" if made is None else made),
                               points=pts, description=desc[:80]))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/Users/ruoqianzhu/Desktop/PBP data")
    ap.add_argument("--output-csv", default="data/intermediate/pbp_events.csv")
    args = ap.parse_args()

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
             if f.lower().endswith((".rtf", ".txt", ".json"))]
    rows: List[dict] = []
    for path in sorted(files):
        content, url = load_content(path)
        gid = game_id_from(path, url)
        if not gid or "actionType" in content and len(content) < 2000:
            print(f"  SKIP {os.path.basename(path)} (no game id / no content)"); continue
        ev = parse_content(content)
        if len(ev) < 50:
            print(f"  SKIP {os.path.basename(path)} (only {len(ev)} events -- likely a summary, not full pbp)"); continue
        for e in ev:
            e["game_id"] = gid
        rows.extend(ev)
        n_made = sum(1 for e in ev if e["action"] in ("FG2", "FG3") and e["made"] == 1)
        print(f"  {gid}: {len(ev)} events, {n_made} made FG, periods {sorted(set(e['period'] for e in ev))}")

    cols = ["game_id", "period", "clock_sec", "person_id", "team_id", "action", "made", "points", "description"]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nWrote {len(rows)} events from {len(set(r['game_id'] for r in rows))} games to {args.output_csv}")


if __name__ == "__main__":
    main()
