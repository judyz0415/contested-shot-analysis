# Methodology (Submission Draft)

We quantify perimeter defensive effectiveness on opponent three-point attempts by combining tracking-derived contest geometry with play-by-play outcomes and a shooter-context baseline model.

---

## Snapshot Metrics

- **Games:** 15
- **Eligible contests (analysis pool):** 509 opponent 3PA rows with Hawk-Eye contest fields and modeling eligibility flags
- **SCQ defender reports:** 12 Heat players as nearest contestant on ≥5 eligible attempts
- **Outcome Label:** Made / Missed (when play-by-play matched)

---

## 1) Data Construction

We build one row per opponent 3PA from Hawk-Eye tracking, then synchronize each shot to NBA play-by-play within a constrained temporal window (same shooter, period, and game-clock compatibility).

| Block | Variables | Source |
|---|---|---|
| Shot identity | game, period, shooter, defender | Hawk-Eye |
| Release + arc | release frame, apex frame, ball trajectory | Hawk-Eye |
| Contest geometry | distance, closeout speed, angle, hand-up | Hawk-Eye |
| Outcome | made/missed, resolution clock, description | NBA play-by-play |

---

## 2) Contest Metric (SCQ)

Shot Contest Quality (SCQ) is a bounded heuristic score on [0,100] computed at release:

`SCQ = 100 x [0.35 * f(distance) + 0.30 * f(speed) + 0.20 * f(angle) + 0.15 * f(hand-up)]`

Each component is clipped to a physically meaningful range. SCQ is interpreted as process quality, not direct outcome probability.

Per-defender SCQ decomposition (means, implied points by driver, deltas vs the eligible-shot pool, scripted coaching cues) is exported as **`data/outputs/heat_defender_scq_breakdown_final.csv`** for Heat defenders with at least five eligible contests.

---

## 3) Baseline vs Full Outcome Models

| Model | Specification | Purpose |
|---|---|---|
| Baseline | `logit(make) ~ shooter skill + shot context` | Estimate expected make probability absent defender identity effects |
| Full | `logit(make) ~ baseline terms + contest terms + defender terms (ridge)` | Capture incremental defender/contest signal with shrinkage |

Shooter skill is proxied by 2025-26 regular-season 3PT%. Shot context includes release/arc geometry and shot clock features.

---

## 4) Defensive Effectiveness Definition

We define defender effectiveness as residual lift relative to baseline expectation:

`lift_vs_baseline = actual_make - baseline_predicted_make`

Defender-level summaries aggregate this residual over contests and apply empirical-Bayes volume normalization:

`volume_normalized_lift = (actual_makes - expected_makes_baseline) / (shots + prior_shots)`

This stabilizes rankings under uneven workload (e.g., 8 vs 57 contests).

---

## 5) Diagnostics and Interpretation

| Diagnostic | Use in analysis |
|---|---|
| Calibration bins | Check probability reliability and model miscalibration regions |
| Brier / logloss | Compare baseline vs full predictive quality |
| SCQ driver decomposition | Translate component-level strengths/deficits into coaching actions |

> **Inference framing**  
> We do not interpret raw FG% allowed as defensive quality. Recommendations are based on context-adjusted residuals and volume-normalized defender effects, supplemented by SCQ component diagnostics for actionable skill development.
