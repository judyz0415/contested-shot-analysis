# Contesting the Three: A Hawk-Eye Analysis of Perimeter Defense

MIT 15.285 Sports Analytics · 2025–26 NBA Season

---

> *"One Heat defender allowed opponents to make only 20.8% of their three-point attempts — nearly 15 percentage points below league average. His box-score metrics ranked him last on his own team's perimeter defense leaderboard. Tracking data tells a different story."*

---

## What This Project Does

This project builds a two-part analytics pipeline on top of **Hawk-Eye optical tracking data** — the 60-fps positional system used by the NBA — to measure *how* Miami Heat defenders contest three-point shots, and what happens as a result.

**15 games · ~3M raw rows/game · 459 Heat-defended 3PA · 6,905 perimeter touch events (both sides)**

---

## Two-Part Analysis

### Part 1 — Shot Contest Quality and Defender Lift

Builds a **Shot Contest Quality (SCQ)** composite score from 6 3D pose features extracted at the release frame:

| Feature | Definition |
|---------|-----------|
| `def_hand_elev_above_ball_in` | Active wrist Z − ball Z at release (inches) |
| `arm_extension_ratio` | dist(shoulder→wrist) / [dist(shoulder→elbow) + dist(elbow→wrist)] |
| `trunk_lean_deg` | Forward lean angle of defender's torso toward shooter |
| `lateral_velocity_ft_s` | Defender 2D centroid speed (15-frame / 0.25s window) |
| `min_def_joint_dist_ft` | Minimum distance from any of 13 defender joints to handler hip |
| `pre_shot_distance_ft` | 2D centroid-to-centroid distance 2s before release |

**Key findings:**
- **Speed paradox:** `corr(pre_shot_distance, closeout_speed) = +0.59` — fast closeouts signal *bad positioning*, not effort. Bam Adebayo has the lowest speed (5.48 ft/s) AND lowest pre-shot distance (11.85 ft) on the team.
- **Noise wall:** Shot outcome AUC ≈ 0.5 (LRT p = 0.795) — 3D pose cannot predict make/miss at n=304. Defense moves the *distribution* of shot quality; outcomes remain noisy.
- **Defender lift:** Bam Adebayo is the only defender with a CI excluding zero (lift = −21.6 pp, CI = [−37.5, −6.4]).

### Part 2 — Action Probability Model

A **Multinomial Logit** model predicting what a ball-handler does when a defender closes out (shoot, drive, pass, catch-and-shoot, foul/turnover) using 3D pose features at the catch frame.

**Feature sets:**
- **F2D-CTX:** `n_open_3pt_threats`, `min_kickout_defender_gap_ft`, `release_shot_clock`, `game_clock_seconds`
- **F3D:** F2D-CTX + 5 pose features above

**Key findings:**
- **LRT F3D vs F2D:** χ²(20) = 207, p = 7×10⁻⁴³ — 3D pose adds significant information beyond court context alone
- **Per-feature LRTs:** All 5 pose features individually significant (min_def_joint_dist strongest at p = 1.2×10⁻⁴³)
- **Calibration:** Weighted ECE = 0.0123 across 5 classes (OOF GroupKFold by game)
- **Sequence model:** Temporal GBM on 12-step pre-catch history scored 0.245 *worse* in log-loss than the snapshot MNLogit — snapshot wins
- **Capstone (closeout tightness → action):** Tight closeouts (< 4 ft) force pass/drive/TO instead of shots at p = 7×10⁻⁶ (n = 3,570 touches) — escaping the make/miss noise wall
- **Counterfactual EPV:** Bam saves +3.31 pts/15 games (374 touches, Δ = −0.008 pts/touch vs. average defender)

---

## Repository Layout

```
contested-shot-analysis/
├── pipeline/                       # Data extraction (raw parquet → cleaned CSVs)
│   ├── load_parquet_from_onedrive.py
│   ├── extract_heat_3pa.py         # Heat 3PA shot extraction
│   ├── detect_make_from_tracking.py # 97%-accurate tracking make-detector
│   ├── build_unified_shot_dataset.py # Canonical Part 1 shot dataset
│   ├── parse_pbp.py                # Play-by-play parsing & action labeling
│   ├── extract_pose_features.py    # 3D pose features at release
│   ├── extract_shot_context_features.py
│   ├── extract_player_states.py
│   ├── extract_closeout_actions.py # Perimeter touch segmentation (Part 2)
│   ├── extract_touch_features.py   # Feature computation at catch frame
│   ├── hawkeye_extract_opponent_3pa.py
│   ├── opponent_three_pointers.py
│   └── load_parquet_from_onedrive.py
│
├── analysis/                       # All analysis scripts
│   ├── part1_shot_quality.py       # Canonical Part 1: SCQ, lift, noise wall
│   ├── action_probability_model.py # Part 2: MNLogit F2D vs F3D, LRT
│   ├── calibrate_action_model.py   # OOF calibration, per-class ECE
│   ├── snapshot_model_diagnostics.py # Hausman-McFadden IIA, VIF
│   ├── sequence_action_model.py    # Temporal GBM comparison
│   ├── per_defender_counterfactual.py # Counterfactual EPV by defender
│   ├── bootstrap_or_ci.py          # OR confidence intervals (forest plot)
│   ├── feature_correlation_analysis.py # Collinearity check (max |r| = 0.463)
│   ├── shot_outcome_wall.py        # AUC noise wall test
│   └── validate_with_pbp.py        # PBP cross-validation
│
├── visualization/                  # Plotting and animation scripts
│   ├── viz.py                      # Core Hawk-Eye 3D renderer
│   ├── shot_viz_from_dataset.py
│   ├── make_gif_paired_comparison.py # GIF pair generation
│   ├── export_scq_extremes.py
│   └── plot_release_snapshot_3d.py
│
├── notebooks/
│   ├── publication_figures.ipynb   # All final paper figures (Figs 1–4)
│   └── action_probability_model.ipynb
│
├── data/
│   ├── PlayerStatistics.csv        # NBA season shooting stats
│   ├── player_height_wingspan.csv  # NBA combine measurements
│   ├── pbp_raw/                    # Raw PBP RTF files (15 games)
│   ├── intermediate/               # Cleaned CSVs (pipeline outputs)
│   │   ├── shot_contest_dataset.csv       # Part 1 input (459 shots)
│   │   ├── touch_events_combined.csv      # Part 2 input (6,905 touches)
│   │   ├── heat_3pa_dataset.csv
│   │   ├── pose_features.csv
│   │   └── ...
│   └── outputs/                    # Analysis results
│       ├── part1_v2/               # Part 1 canonical outputs
│       ├── action_probability_3d.csv
│       ├── action_probability_per_feature_lrt.csv
│       ├── action_model_calibration.csv
│       ├── counterfactual_defender_values_heat_only.csv
│       ├── hausman_mcfadden.csv
│       ├── figures/                # Publication figures (PNGs + GIFs)
│       └── plots/
│
├── report/
│   ├── heat_perimeter_defense_report.md
│   ├── Heat_Perimeter_Defense_Report_Final.docx
│   ├── Heat_Perimeter_Defense_Technical_Report.docx
│   └── assets/                     # SCQ extreme GIFs and metric cards
│
├── teamworks_project_spec_v2.md    # Full project specification
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/<you>/contested-shot-analysis
cd contested-shot-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The intermediate CSVs are tracked in the repo. Analysis scripts run without raw Hawk-Eye parquet files (which are ~1.1 GB/game and not vendored).

## Reproducing Results

**Part 1 — SCQ and defender lift:**
```bash
python analysis/part1_shot_quality.py
# Outputs → data/outputs/part1_v2/
```

**Part 2 — Action probability model:**
```bash
python analysis/action_probability_model.py --touches data/intermediate/touch_events_combined.csv
python analysis/calibrate_action_model.py
python analysis/snapshot_model_diagnostics.py
python analysis/sequence_action_model.py
python analysis/per_defender_counterfactual.py
python analysis/bootstrap_or_ci.py
```

**Publication figures:**
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/publication_figures.ipynb
```

---

## Data Notes

Raw Hawk-Eye parquets are stored externally (OneDrive). Each game file has ~3M rows × 111 columns at 60fps. The pipeline filters to ~170k live-play frames, then extracts ~460 perimeter touch events per game. See [`pipeline/load_parquet_from_onedrive.py`](pipeline/load_parquet_from_onedrive.py) for the loader.

---

*Developed for MIT 15.285 (Sports Analytics) using Hawk-Eye data provided through the course.*
