# 15.285-ContestedProject

## Load parquet files from local OneDrive

This project includes a starter script to read parquet files from:

`/Users/mariaangellobon/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025`

### Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run:
   `python load_parquet_from_onedrive.py`

Optional arguments:

- `--folder`: override the default folder path.
- `--sample-rows`: number of preview rows from the first file (default: 5).

## Notes

- Be careful about overclaiming on defender intent — the data provides imputed intent at best
- When presenting SCQ, always include scale context (min, max, central tendency)
- Always activate the project venv (`source /Users/mariaangellobon/Desktop/TrackingData/.venv/bin/activate`) — system anaconda has a NumPy version conflict

## Next Steps

### In Progress
- [ ] **Merge PBP shot outcome into feature CSV** — `opponent_3pa_events_with_contest_features.csv` has no `shot_result` (made/missed) column. Run `opponent_three_pointers.py` across all 15 games and join on game + period + game clock to attach shot outcome.
- [ ] **Build per-shot time windows** — for each shot event, extract the full tracking window from 5 seconds before ball release to shot resolution (make/miss). Store as a structured record per shot for downstream analysis and visualization.
- [ ] **Dual-defender detection** — extend the nearest-defender logic to flag a second Heat defender if they are within a similar distance to the primary (threshold TBD, ~2 ft). Add `primary_defender`, `secondary_defender`, `dual_closeout_flag` columns.
- [ ] **Rename time-source columns for clarity** — `frame`/`game_clock` currently refer to the Hawk-Eye release event. Add clearly named columns for PBP resolution time and Hawk-Eye release time to avoid confusion.

### Pending Decisions
- [ ] **Define target variable for modeling** — candidates: binary `shot_made`, expected FG% delta, or shot difficulty score. This determines what SCQ ultimately predicts and how to interpret it.
- [ ] **Dual-closeout distance threshold** — decide how close a second defender must be to the primary to count as a co-contestant.
- [ ] **Corner vs. above-the-break stratification** — these are structurally different shots; decide whether SCQ should be split by zone.

### Data Cleaning
- [ ] **Garbage time filtering** — standard definition: large lead (e.g., >15 pts) + late game (Q4, <5 min remaining). Apply before any outcome-based analysis so garbage-time defensive effort doesn't bias the metric.
- [ ] **False-positive shot detection QA** — validate Hawk-Eye ball-z candidate counts against PBP confirmed 3PA counts per game. Some candidates are dribble bumps or other ball elevation events.

### Future (Modeling Phase)
- [ ] Logistic regression / XGBoost on shot outcome using contest features
- [ ] Evaluate whether mixed-effects models are needed to control for shooter or defender identity
- [ ] Defender normalization — hand-up height may need to be normalized by player height
