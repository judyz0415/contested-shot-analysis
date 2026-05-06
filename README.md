# 15.285-ContestedProject

**Shot Contest Quality** — a tracking-based defensive metric built from Hawk-Eye data for 15 Miami Heat games (2024–25 season).

## Repository Structure

```
15.285-ContestedProject/
├── scripts/
│   ├── build_unified_shot_dataset.py     ← main pipeline (run this)
│   ├── opponent_three_pointers.py        ← PBP sync utility
│   ├── load_parquet_from_onedrive.py     ← parquet preview utility
│   └── hawkeye_extract_opponent_3pa.py   ← superseded; kept for reference
├── data/
│   ├── outputs/                          ← generated locally (gitignored — do not commit CSVs)
│   └── archive/                          ← old local extracts (gitignored)
├── requirements.txt
└── README.md
```

Hawk-Eye parquet files live on OneDrive (not in this repo). Example path (quote it in the shell):

`/Users/mariaangellobon/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025`

## Quick Start

1. Activate the virtual environment:
   ```bash
   source /Users/mariaangellobon/Desktop/TrackingData/.venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main pipeline (writes the three CSVs under `data/outputs/` when you point `--output-csv` there):
   ```bash
   python scripts/build_unified_shot_dataset.py \
       --input-dir '/Users/mariaangellobon/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025' \
       --output-csv data/outputs/shot_contest_dataset.csv
   ```

Do **not** use placeholder paths like `/path/to/miami_heat_2025` — the script requires a real directory containing `*_processed.parquet` files.

## Pipeline decisions (unified build)

The script runs in two stages for each game:

1. **Primary Hawk-Eye pass** — Same as before: upward `ball_z` crossing at ~7.5 ft (90 in), opponent shooter, three-point distance, ball approaches a rim within a tight proximity check, nearest Heat defender at release, then PBP match on shooter + period + game clock (~5 s window after release).

2. **PBP rescue pass (on by default)** — For opponent 3PA rows that never matched tracking, search again using the **PBP resolution clock** as an anchor: tracking **release** should sit about **1–2 seconds earlier** on the **game-clock countdown** than the PBP timestamp (configurable lag window). Uses the same shooter (`last_touch`), a **relaxed** ball-height crossing and a **looser** rim-proximity check so flat shots and some airballs can still yield a row. Successful rows are tagged **`pbp_rescued` = yes** and linked to that PBP event. This reduces back-to-back same-shooter misses (e.g. two pull-ups in a few seconds) when the primary matcher attached the wrong PBP row or missed one release.

**Analytic inclusion** is separate from raw detection:

- **`analysis_eligible`** — `yes` only if the row passes all filters below; use this for modeling or summary tables that should exclude junk or ambiguous events.
- **`exclusion_reason`** — Semicolon-separated codes when `analysis_eligible` is `no`.

Exclusion rules (tune via CLI):

| Code | Meaning |
|------|---------|
| `no_pbp_match` | Tracking detection with no matching PBP (treated as out of scope for outcome-based analysis; after video review many are post-whistle or non-shot motion). |
| `heave_long_distance` | Shooter distance to the **attacking** rim ≥ `--heave-min-ft-from-rim` (default **42 ft**, ~half-court territory). |
| `shot_clock_below_1s` | Shot clock at release strictly **&lt; 1.0 s**. |
| `desperate_shot_clock_le_0p8` | Shot clock at release **≤ 0.8 s** (explicit desperation tag). |
| `pbp_keyword_heave` | `pbp_description` contains `heave`. |

**`suspected_low_arc_or_l`** — Heuristic only: apex ball height below ~**158 in** suggests a flat trajectory or a mis-tagged play (e.g. lob near the rim). Does **not** auto-exclude; use for manual QA.

## CLI reference (`build_unified_shot_dataset.py`)

| Flag | Default | Role |
|------|---------|------|
| `--input-dir` | (required) | Folder of `*_processed.parquet` |
| `--output-csv` | `shot_contest_dataset.csv` | Main output; `_unmatched` and `_excluded_heaves` paths are derived from this basename unless overridden |
| `--pbp-delay` | `0.5` | Sleep between NBA CDN PBP requests |
| `--no-pbp-rescue` | off | Disable the second-pass PBP rescue |
| `--pbp-rescue-lag-min` / `--pbp-rescue-lag-max` | `0.9` / `2.8` | Game-clock seconds: expected release remaining minus PBP remaining |
| `--pbp-rescue-preferred-lag` | `1.55` | Preferred offset inside that window |
| `--heave-min-ft-from-rim` | `42` | Long-distance heave cutoff (feet from rim) |
| `--min-shot-clock-analysis` | `1.0` | Minimum release shot clock to stay eligible |
| `--desperate-shot-clock` | `0.8` | Also flag ≤ this value in `exclusion_reason` |
| `--excluded-heaves-csv` | auto | Override path for the heave audit file |

## Output Files

### `shot_contest_dataset.csv` — primary dataset

One row per **tracking-backed** opponent 3-point attempt (primary pass plus any **rescued** PBP row that found a matching release). Counts change when you re-run the pipeline; older bullet statistics in this README may be stale—recompute from your CSV after each run.

Column groups:

| Block | Columns | Source |
|------|---------|--------|
| Identifiers | `game_file`, `game_id`, `opponent`, `period` | — |
| Release | `release_frame`, `release_game_clock`, `release_shot_clock`, shooter info, ball at release | Hawk-Eye |
| Arc apex | `apex_frame`, `apex_game_clock`, `apex_shot_clock`, `apex_ball_z` | Hawk-Eye |
| Contest | `min_ball_rim_3d_in`, defender info, `contest_*`, `hand_up_in`, `shot_contest_quality` | Hawk-Eye |
| PBP | `pbp_shot_result`, `pbp_conclusion_game_clock`, `pbp_description` | NBA CDN |
| QA / filters | `pbp_rescued`, `analysis_eligible`, `exclusion_reason`, `suspected_low_arc_or_lob` | Derived |

Clock semantics:

- **`release_game_clock`** — When the ball leaves the shooter’s hands (tracking).
- **`pbp_conclusion_game_clock`** — When the play-by-play records the shot resolving (made/missed); typically **earlier** on the countdown than release by about **1–2 s** of game time.

Generated by: `scripts/build_unified_shot_dataset.py`

---

### `shot_contest_dataset_unmatched.csv` — diagnostic file

Still produced every run. Rows where Hawk-Eye and PBP disagree **after** rescue:

- **`tracking_only`** — Release detected but no PBP match. For analysis we mark these **`analysis_eligible` = no** in the main file (`no_pbp_match`). Video review on a sample game suggests many are fouls / post-whistle motion, not countable field-goal tries.
- **`pbp_only`** — PBP lists a 3PA that never linked to tracking even after rescue (very flat trajectory, missing `last_touch`, data gap, etc.). Outcome-only for audit; no contest features.

Generated by: `scripts/build_unified_shot_dataset.py` (same run)

---

### `shot_contest_dataset_excluded_heaves.csv` — heave / desperation audit

Subset of main-dataset rows whose **`exclusion_reason`** includes any of:

`heave_long_distance`, `shot_clock_below_1s`, `desperate_shot_clock_le_0p8`, `pbp_keyword_heave`

Rows excluded **only** for `no_pbp_match` are **not** listed here (they remain in the main CSV with `analysis_eligible` = no).

Override output path with `--excluded-heaves-csv` if needed.

---

## Notes

- Be careful about overclaiming on defender intent — the data provides imputed intent at best.
- When presenting SCQ, always include scale context (min, max, central tendency).
- Always activate the project venv (`source /Users/mariaangellobon/Desktop/TrackingData/.venv/bin/activate`) — system Anaconda has a NumPy version conflict with some stacks.

## Next Steps

### In Progress / Recently Addressed

- [x] **Merge PBP shot outcome into feature CSV** — `shot_contest_dataset.csv` via `build_unified_shot_dataset.py`
- [x] **Rename time-source columns** — `release_game_clock`, `apex_game_clock`, `pbp_conclusion_game_clock`
- [x] **Recover many `pbp_only` events** — second-pass clock-anchored rescue + relaxed geometry; see `pbp_rescued`
- [x] **Analytic filters** — `analysis_eligible`, heave / shot-clock exclusions, `_excluded_heaves.csv` audit
- [x] **Low-arc / possible lob flag** — `suspected_low_arc_or_l` for manual review
- [ ] **Continued unmatched audit** — especially remaining `pbp_only` and video validation of `suspected_low_arc_or_lob`
- [ ] **Shot segment table** — lightweight table with `pre_release_start_frame`, `release_frame`, `apex_frame`, `conclusion_frame` (conclusion frame requires parquet clock→frame lookup; see `opponent_three_pointers.py` for precedent)
- [ ] **Dual-defender detection** — secondary Heat defender within ~2 ft of primary; `dual_closeout_flag`

### Pending Decisions

- [ ] **Target variable for modeling** — binary `shot_made`, expected FG% delta, or shot difficulty score
- [ ] **Dual-closeout distance threshold**
- [ ] **Corner vs. above-the-break stratification**

### Data Cleaning

- [ ] **Garbage time filtering** — e.g. large lead + Q4 &lt; 5 min before outcome-based analysis
- [ ] **False-positive shot detection QA** — compare tracking counts to PBP 3PA per game where `analysis_eligible` is yes

### Future (Modeling Phase)

- [ ] Logistic regression / XGBoost on shot outcome using contest features among `analysis_eligible` rows
- [ ] Mixed-effects models for shooter/defender identity
- [ ] Defender normalization (e.g. hand-up height vs. player height)
