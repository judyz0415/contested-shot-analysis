# Data cleaning pipeline: unified shot contest dataset

This document describes how **`data/intermediate/shot_contest_dataset.csv`** is produced by **`scripts/pipeline/build_unified_shot_dataset.py`**, including what gets filtered at each stage and row counts from the last logged run (`data/intermediate/build_unified_shot_dataset.log`, 15 Heat home games).

---

## 1. Outputs (what “clean” means here)

| Artifact | Rows (last run) | Role |
|----------|----------------:|------|
| **`shot_contest_dataset.csv`** | **561** | Main table: one row per **Hawk-Eye–detected** opponent three-point release that passes tracking geometry checks, with contest features and (when possible) NBA play-by-play outcome. |
| **`shot_contest_dataset_unmatched.csv`** | **90** | Audit file for alignment gaps (**not** extra shots): **51** tracking rows without PBP + **39** PBP 3PA events with no tracking row. |
| **`shot_contest_dataset_excluded_heaves.csv`** | **4** | Subset of main rows whose `exclusion_reason` includes **heave / desperation / long-distance** codes (see §6). |

**Analysis-ready subset:** **`analysis_eligible == yes`** → **509 rows** (see §6).

Rows are **not** dropped from the main CSV when they fail eligibility; they stay in the file with `analysis_eligible=no` and an `exclusion_reason` code so you can audit losses.

---

## 2. Inputs

1. **Processed Hawk-Eye parquet files** (`*_processed.parquet`) under `--input-dir`: ball track, `last_touch_player_id`, clocks, player positions, wrist heights, etc.
2. **NBA play-by-play JSON** from the CDN (`playbyplay_{game_id}.json`), fetched per game.

Anything that never becomes a tracking row never enters the CSV (§4).

---

## 3. Play-by-play: define opponent 3PA events

For each game, actions are filtered to **opponent three-point field goal attempts**:

- Field goal actions only (`isFieldGoal`).
- Description contains **`3PT`** or **`3-PT`**.
- **`teamId` ≠ Miami Heat** (`1610612748`).
- Valid **`personId`**.

**Last run:** **549** such PBP events across 15 games (sum of per-game counts in the log).

These events are the **target list** for matching; they are **not** the row count of the final dataset.

---

## 4. Tracking: from frames to candidate shots (implicit drops)

The script scans video frames (spacing rules avoid duplicate releases). For each index `i`, **`_build_row_for_release_index`** either builds a row or returns **`None`**. There is **no central counter** of rejected candidates in the log; rejections are **per-rule**:

| Gate | Purpose |
|------|--------|
| **Release geometry** | Ball height crosses `RELEASE_Z_THRESHOLD` (90 in) upward with continued rise (primary pass); rescue pass uses relaxed tiers (e.g. 78 in, 72 in). |
| **`last_touch`** | Must identify a shooter at release. |
| **Shooter team** | Must **not** be Miami → opponent shot only. |
| **Rim approach** | Minimum 3D ball–rim distance in a short window after release ≤ **18 in** (primary); rescue uses **38 in** for difficult/airball paths. |
| **Shooter position** | Shooter centroid required at release frame. |
| **Beyond-the-arc** | Shooter rim distance ≥ **`THREEPT_MIN_INCHES`** (264 in ≈ 22 ft). |
| **Heat defender** | Nearest Miami player to shooter at release must exist (for contest geometry). |
| **Apex window** | Enough frames after release to define arc apex (`APEX_SEARCH_FRAMES`). |

**Last run:** **561** tracking rows survived these gates (sum of per-game “Tracking rows” in the log).

All **561** are written to **`shot_contest_dataset.csv`**.

---

## 5. Linking tracking rows to PBP (made / missed)

### Primary match

- Match on **period**, **shooter `personId`**, and PBP clock within **`[release_remaining − 5 s, release_remaining + 1 s]`** (clock counts down).
- Closest event in time wins.

### Rescue pass (`pbp_rescued`)

For PBP events still unmatched, a **second pass** searches release frames in a **clock lag window** around the PBP **resolution** time (typical flight-time offset), with relaxed release-height tiers and looser rim proximity (**`RESCUE_RIM_APPROACH_MAX_IN`**). Successful rescues set **`pbp_rescued=yes`**.

**Last run (main CSV):**

| Metric | Count |
|--------|------:|
| Rows with **Made/Miss** (`pbp_shot_result`) | **510** |
| Rows with **no PBP outcome** (`pbp_shot_result` missing) | **51** |
| Rows with **`pbp_rescued=yes`** | **102** (subset of PBP-matched rows aligned via the rescue pass; sum of per-game counts in the log) |

The **51** no-PBP rows are **still in the main file** (contest features present; outcome missing).

---

## 6. QA flags: `analysis_eligible` and `exclusion_reason`

After each row is built, **`_finalize_row_analysis_columns`** sets eligibility.

**`analysis_eligible=no`** if **any** of:

| Code | Rule |
|------|------|
| **`no_pbp_match`** | No matched PBP Made/Miss for that row. |
| **`shot_clock_below_1s`** | Release shot clock **strictly &lt; 1.0 s** (after parse). |
| **`desperate_shot_clock_le_0p8`** | Shot clock **≤ 0.8 s** (also flagged as desperation). |
| **`heave_long_distance`** | Shooter distance to attacking rim **≥ 42 ft** (default CLI), converted to inches in code. |
| **`pbp_keyword_heave`** | `"heave"` appears in matched PBP description (lowercase check). |

If none apply → **`analysis_eligible=yes`**, **`exclusion_reason`** empty.

**Separate QA column:** **`suspected_low_arc_or_lob`** (`yes` if apex ball height &lt; **`LOW_ARC_APEX_Z_INCHES`**). This **does not** by itself flip `analysis_eligible`; it’s a diagnostic for possible lob/mis-tag.

### Row counts (verified on `shot_contest_dataset.csv`)

| Category | Rows |
|----------|-----:|
| **`analysis_eligible=yes`** | **509** |
| **`analysis_eligible=no`** | **52** |

**Breakdown of `exclusion_reason` (52 rows):**

| `exclusion_reason` | Rows |
|--------------------|-----:|
| `no_pbp_match` | 48 |
| `no_pbp_match;heave_long_distance` | 3 |
| `shot_clock_below_1s;desperate_shot_clock_le_0p8` | 1 |

Notes:

- All **51** rows without PBP outcome are ineligible (**`no_pbp_match`** appears on each; three also tag **`heave_long_distance`**).
- **One** row **did** match PBP (**Made**) but was excluded for **sub-1s shot clock** / desperation — so **510 − 1 = 509** eligible with outcomes.

### Defender model eligibility (`defender_model_eligible`)

After all games are merged, the builder assigns **`shooter_2025_26_regular_3pt_pct`** from **`--player-statistics-csv`** (default `data/PlayerStatistics.csv`) using the same shrinkage logic as `model_defensive_effectiveness.py`. If that file is missing, an empty prior table with **league_pct = 0.36** is used (every known `shooter_id` still gets a finite prior).

**`defender_model_eligible=yes`** only when **`analysis_eligible=yes`** and **all** lift-model inputs are usable: non-empty nearest defender and shooter ids, PBP result contains `made` or `miss`, finite **`shooter_2025_26_regular_3pt_pct`**, **`shooter_dist_to_rim_in`**, **`release_ball_z`**, **`apex_ball_z`**, parseable finite **`release_shot_clock`**, and finite contest numerics including **`closeout_delta_ft_500ms`** and **`shot_contest_quality`**. Otherwise **`defender_model_eligible=no`** with a **`defender_model_exclusion_reason`** code (e.g. `analysis_ineligible`, `nonfinite_closeout_delta_ft_500ms`).

Downstream, pass **`--defender-model-eligible-only`** to **`model_defensive_effectiveness.py`** and **`explain_scq_drivers_by_defender.py`** so defender lift and SCQ driver tables use the **same** shot rows. Re-run the builder after changing tracking or PBP logic so these columns stay in sync.

### Excluded-heaves audit file

**`shot_contest_dataset_excluded_heaves.csv`** (4 rows) lists main-file rows where **`exclusion_reason`** intersects **`heave_long_distance`**, **`shot_clock_below_1s`**, **`desperate_shot_clock_le_0p8`**, or **`pbp_keyword_heave`**. Rows that are ineligible **only** due to **`no_pbp_match`** are **not** copied there.

---

## 7. Unmatched sidecar (alignment audit, not dropped from main)

**`shot_contest_dataset_unmatched.csv`** (**90** rows):

| `unmatched_type` | Rows | Meaning |
|------------------|-----:|---------|
| **`tracking_only`** | **51** | Same attempts as the **51** main rows with missing `pbp_shot_result`; duplicated in simplified form for review. |
| **`pbp_only`** | **39** | PBP opponent 3PA with **no** tracking row matched (blocked shots, sync misses, etc.). |

Accounting check: **549** PBP events ≈ **510** matched to tracking + **39** `pbp_only`.

---

## 8. End-to-end summary (last run)

```
PBP opponent 3PA events (15 games)     549
        └─ matched to tracking          510  → appear in main CSV with Made/Miss
        └─ never matched (pbp_only)      39  → unmatched CSV only

Hawk-Eye tracking rows written          561  → main CSV (all)
        └─ with PBP outcome               510
        └─ without PBP outcome             51  → ineligible (no_pbp_match)

analysis_eligible=yes                     509  ← typical “clean modeling” subset
```

---

## 9. Reproducing counts

Re-run the builder (with your parquet directory and SSL settings as needed):

```bash
python scripts/pipeline/build_unified_shot_dataset.py \
  --input-dir /path/to/parquets \
  --output-csv data/intermediate/shot_contest_dataset.csv \
  --excluded-heaves-csv data/intermediate/shot_contest_dataset_excluded_heaves.csv
```

The console summary prints per-game tracking vs matched vs eligible totals and the aggregate numbers in §8.

---

## 10. Downstream analysis (aligned row pool)

The builder now tags **`defender_model_eligible`** using the same finite-field and outcome rules as **`model_defensive_effectiveness.py`**. Pass **`--defender-model-eligible-only`** there and in **`explain_scq_drivers_by_defender.py`** so defender lift and SCQ breakdowns share one shot pool without silent drops inside each script. Older CSVs without this column still rely on per-script filtering.
