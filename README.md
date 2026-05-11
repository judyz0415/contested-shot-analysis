# Shot Contest Quality — Hawk-Eye Pipeline

Python tooling that aligns Hawk-Eye optical tracking with NBA play-by-play for opponent three-point attempts (Miami Heat home measurements, 2024–25 season). Each detection yields release timing, arc apex, defender proximity and kinematics, and a composite contest-quality score, merged with official shot outcome when matched.

MIT Sloan **Analytics Insights (15.285)** project. Processed Hawk-Eye parquet files stay outside this repo.

## Layout

```
├── scripts/
│   ├── pipeline/                       # Dataset construction + matching
│   │   ├── build_unified_shot_dataset.py
│   │   ├── opponent_three_pointers.py
│   │   ├── load_parquet_from_onedrive.py
│   │   └── hawkeye_extract_opponent_3pa.py
│   ├── visualization/                  # 3D rendering + animations
│   │   ├── viz.py
│   │   ├── shot_viz_from_dataset.py
│   │   └── plot_release_snapshot_3d.py
│   └── analysis/                       # Modeling and effect analysis
│       ├── model_defensive_effectiveness.py
│       ├── visualize_defensive_effectiveness.py
│       └── explain_scq_drivers_by_defender.py
├── notebooks/
│   ├── vizualisation.ipynb
│   └── shot_viz_from_dataset.ipynb
├── data/outputs/                       # Local generated artifacts (gitignored)
├── requirements.txt
└── README.md
```

Processed parquet games are not vendored in this repository. Point `--input-dir` at a folder that contains `*_processed.parquet`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/pipeline/build_unified_shot_dataset.py \
  --input-dir /path/to/processed_parquets \
  --output-csv data/outputs/shot_contest_dataset.csv
```

The merge script pulls play-by-play from the NBA CDN (`cdn.nba.com`) using each game’s ID from the filename. TLS verification uses the **`truststore`** package (OS trust store, e.g. macOS Keychain) with fallbacks to **certifi** and the default SSL context. Run `pip install -r requirements.txt` so PBP fetches verify correctly. Use `--insecure-pbp-ssl` only as an emergency workaround.

## Methodology

**Detection.** Candidate releases use an upward `ball_z` crossing (~90 in), opponent possession (`last_touch_player_id`), arc motion toward a rim, shooter beyond the NBA three-point distance in Hawk-Eye coordinates, and the nearest Miami defender at release for contest geometry.

**Primary PBP link.** Each detection matches an opponent 3PA event on shooter ID, period, and game clock within a short window around release.

**Rescue pass.** Events that appear in PBP but miss on the first pass are searched again: frames where clock and `last_touch` match the PBP shooter and release falls on the remaining clock slightly before the logged resolution time (configurable lag). Thresholds for ball height and rim proximity are relaxed to recover flat or missed-track releases.

**Filtering.** `analysis_eligible` summarizes rows suitable for modeling; `exclusion_reason` lists structured codes (missing PBP match, long-distance heave, low shot clock at release, “heave” in description). `suspected_low_arc_or_lob` flags unusually low apex height for manual review only.

## Outputs

| Artifact | Description |
|----------|-------------|
| `shot_contest_dataset.csv` | One row per tracking-backed attempt with contest features and PBP fields |
| `shot_contest_dataset_unmatched.csv` | `tracking_only` vs `pbp_only` diagnostics after rescue |
| `shot_contest_dataset_excluded_heaves.csv` | Rows excluded primarily for distance or clock rules |

Companion filenames derive from `--output-csv` unless overridden (`--excluded-heaves-csv`).

## CLI (main script)

| Flag | Purpose |
|------|---------|
| `--input-dir` | Directory of `*_processed.parquet` files (required) |
| `--output-csv` | Primary CSV path |
| `--pbp-delay` | Pause between CDN requests (seconds) |
| `--no-pbp-rescue` | Skip second-pass PBP alignment |
| `--pbp-rescue-lag-min`, `--pbp-rescue-lag-max`, `--pbp-rescue-preferred-lag` | Expected release vs resolution clock offset (seconds) |
| `--heave-min-ft-from-rim` | Exclude attempts beyond this arc distance from the attacking rim |
| `--min-shot-clock-analysis`, `--desperate-shot-clock` | Shot-clock thresholds at release |
| `--excluded-heaves-csv` | Custom path for heave audit extraction |
| `--insecure-pbp-ssl` | Emergency only: disable TLS verification for PBP (avoid if possible) |

## Visualization Workflow

Generate release PNG + rotatable HTMLs from shot rows:

```bash
python scripts/visualization/shot_viz_from_dataset.py \
  --shots-csv data/outputs/shot_contest_dataset.csv \
  --parquet-dir /path/to/processed_parquets \
  --row-indices 150 \
  --pre-frames 120 \
  --post-frames 6
```

Outputs are written to `data/outputs/shot_visualizations/<shot_id>/` with:
- `release_snapshot_court.png`
- `release_interactive.html`
- `pre_release_animation.html`
- `window_animation.html`

## Schema (high level)

Identifiers and clocks (`release_*`, `apex_*`), ball position at release, rim alignment (`rim_x`, `rim_y`), `min_ball_rim_3d_in`, nearest defender identity and distance, closeout speed and angle, `hand_up_in`, `shot_contest_quality`, then PBP outcome columns plus `pbp_rescued`, `analysis_eligible`, `exclusion_reason`, `suspected_low_arc_or_lob`.

Release clock reflects ball leaving the hands; PBP conclusion clock reflects when the feed logs resolution (typically a short interval earlier on the game countdown).

Optional scratchpad: create `LOCAL_NOTES.md` in the repo root for reminders and backlog (that filename is gitignored).
