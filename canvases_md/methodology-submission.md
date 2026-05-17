# Methodology (Submission Draft)

We quantify perimeter defensive effectiveness on opponent three-point attempts by combining tracking-derived contest geometry with play-by-play outcomes and a shooter-context baseline model.

---

## Snapshot Metrics

- **Games:** 15
- **Tracked 3PA (geometry filters passed):** 459
- **Modeled shots (complete features + matched outcome):** 399
- **SCQ defender reports:** 12 Heat players as nearest contestant on ≥5 eligible attempts
- **Outcome label:** Made / Missed (from NBA play-by-play)

*Note: 51 of 459 tracked shots have no matched PBP outcome (tracking-only); 9 have missing shot-clock readings. Both categories are excluded from outcome modeling but retained in the SCQ process analysis.*

---

## 1) Data Construction

We build one row per opponent 3PA from Hawk-Eye tracking, then synchronize each shot to NBA play-by-play within a constrained temporal window (same shooter, period, and game-clock compatibility).

| Block | Variables | Source |
|---|---|---|
| Shot identity | game, period, shooter, defender | Hawk-Eye |
| Release + arc | release frame, apex frame, ball trajectory | Hawk-Eye |
| Contest geometry | distance, closeout speed, angle, hand-up | Hawk-Eye |
| Outcome | made/missed, resolution clock, description | NBA play-by-play |

Of 549 PBP opponent 3PA events, 510 matched to a Hawk-Eye tracking row. Geometry gates (arc threshold, rim-approach confirmation, beyond-the-arc position) reduced that to 459 rows; 51 heave/desperation-clock rows are flagged ineligible. The final modeling pool is **399 shots** with complete features and a matched Made/Missed result.

---

## 2) Contest Metric (SCQ)

Shot Contest Quality (SCQ) is a bounded heuristic score on [0, 100] computed at release:

`SCQ = 100 × [0.35 × f(distance) + 0.30 × f(speed) + 0.20 × f(angle) + 0.15 × f(hand-up)]`

Each component is clipped to a physically meaningful range. Weights are expert-specified heuristics, not data-derived. SCQ is interpreted as **process quality**, not direct outcome probability. It is used for coaching decomposition (Section 6) but excluded from the outcome model's contest features to avoid near-perfect collinearity with its own components.

---

## 3) Baseline vs Full Outcome Models

| Model | Specification | Purpose |
|---|---|---|
| Baseline | `logit(make) ~ shooter_skill + shot_context` | Estimate expected make probability absent defender identity effects |
| Full | `logit(make) ~ baseline_terms + contest_terms + defender_fixed_effects (ridge)` | Capture incremental defender and contest signal with L2 shrinkage |

**Shooter skill** is proxied by 2025-26 regular-season 3PT% (Beta-shrunk toward the league mean using 50 pseudo-attempts). Because the same season encompasses these 15 games and roughly 1,215 other regular-season games, the circularity is small but non-zero; a prior-season proxy would be cleaner and is preferred in future iterations.

**Contest features in the full model:** `contest_distance_ft`, `closeout_speed_ft_s`, `contest_angle_deg`, `hand_up_in`. Two variables present in the raw data are deliberately excluded: `closeout_delta_ft_500ms` is excluded because it equals `closeout_speed_ft_s × 0.5` (perfect collinearity). `shot_contest_quality` is excluded because it is a weighted linear combination of the four included components, introducing near-perfect collinearity that prevents stable coefficient estimation.

**Defender terms** are one-hot encoded fixed-effect indicators (one defender dropped as reference), regularized with L2 penalty (λ = 6.0). This approximates partial pooling but is not a hierarchical model; coefficients represent deviation from the omitted reference defender.

---

## 4) Defensive Effectiveness Definition

We define defender effectiveness as residual lift relative to baseline expectation:

`lift_vs_baseline = actual_make − baseline_predicted_make`

Defender-level summaries aggregate this residual over contests and apply empirical-Bayes volume normalization:

`volume_normalized_lift = (actual_makes − expected_makes_baseline) / (shots + 25)`

This stabilizes rankings under uneven workload (e.g., 8 vs. 57 contests) by shrinking each defender's observed lift toward zero at weight `n / (n + 25)`. A defender with 8 shots is trusted at 24% of face value; one with 57 shots is trusted at 70%.

---

## 5) Diagnostics and Interpretation

| Diagnostic | Use in analysis |
|---|---|
| Calibration bins | Check probability reliability and model miscalibration regions |
| Brier / logloss | Compare baseline vs. full predictive quality |
| SCQ driver decomposition | Translate component-level strengths/deficits into coaching actions |

**Baseline discrimination caveat:** The baseline model's predicted probabilities span 0.20–0.60, concentrated near 0.45–0.55 for most shots. This narrow range reflects the limited shooter-to-shooter variation in our 15-game sample and the inherent noise in single-attempt outcomes. As a result, lift vs. baseline retains residual shot-context noise. Recommendations are therefore based primarily on SCQ component diagnostics for actionable skill development, with lift used as a directional outcome signal rather than a precise causal estimate.

> **Inference framing:** We do not interpret raw FG% allowed as defensive quality. Recommendations are based on context-adjusted residuals and volume-normalized defender effects, supplemented by SCQ component diagnostics.

---

## 6) Tracking Snapshots — SCQ in Practice

The four captures below are drawn from game 0022500062 (Heat vs. Bucks). Each shows the Hawk-Eye skeleton at the release frame: the shooter appears in **green**, the nearest Heat defender in **blue**, and the ball at release as a **red dot**. Together they illustrate the range of contest quality the metric is designed to detect.

---

### Figure 1 — Frame 1146 | Ryan Rollins vs. Norman Powell | SCQ: 17.36 | Result: **Missed**

![Frame 1146 — poor contest, Norman Powell far from shooter](../data/outputs/visualizations/report/game 0022500062 | frame 1146 | Ryan Rollins vs defender 1626181.png)

Despite the miss, this represents a defensive breakdown. Powell was 9.3 ft from Rollins at release — near the 10 ft cap where the distance component yields zero credit — and was moving **away** from the shooter at −2.1 ft/s. Negative closeout speed means Powell was retreating or rotating out of the play as Rollins elevated. The contest angle (168°) falls beyond the formula's 90° ceiling, contributing zero. The only non-zero contributor is hand height (17.8 in), likely a trailing arm. The visualization confirms it: Powell's skeleton (blue) sits well off the arc while Rollins (green) rises near the right wing. The ball went in the basket for the miss due to shooter execution, not defensive deterrence — exactly the type of empty "contest" this metric is designed to flag.

---

### Figure 2 — Frame 18257 | Ryan Rollins vs. Bam Adebayo | SCQ: 48.63 | Result: **Missed**

![Frame 18257 — moderate contest, Bam Adebayo with high hands](../data/outputs/visualizations/report/game 0022500062 | frame 18257 | Ryan Rollins vs defender 1628389.png)

Adebayo arrived at 6.5 ft with a closing speed of 5.7 ft/s — solid but not elite. His defining contribution is vertical reach: at 44.6 inches above his body centroid (nearly 3.7 feet), his hand height reaches full formula credit. The skeleton confirms an upright posture with arms extended toward the ball path. The shot missed. This contest also illustrates a known SCQ limitation: Adebayo's length extends his deterrent radius beyond what proximity alone measures. A 6.5 ft contest by a 6'9" center with a 7'3" wingspan is meaningfully different from the same geometry for a 6'1" guard — a distinction the current formula does not capture by design. Future versions that measure hand position relative to ball position at release would close this gap.

---

### Figure 3 — Frame 3966 | Ryan Rollins vs. Davion Mitchell | SCQ: 69.61 | Result: **Missed**

![Frame 3966 — strong contest, Mitchell at full closeout speed with tight proximity](../data/outputs/visualizations/report/game 0022500062 | frame 3966.png)

This is the highest-SCQ contest in the sample. Mitchell closed to 3.0 ft — barely a step — at 8.4 ft/s, hitting the formula's speed ceiling. His hand height reached 45.9 in. The image shows the two skeletons nearly overlapping at the arc. Distance and speed both contribute close to their maximum weights; the high score is earned almost entirely through exceptional proximity and urgency of closeout. The angle (165°) falls outside the formula's credit range, a reminder that the angle component contributes zero across all four examples here — a calibration limitation addressed in the paper's limitations section. The shot missed. This frame is the archetype of a strong perimeter contest: a full-speed closeout that terminates in tight, hand-up proximity.

---

### Figure 4 — Frame 5724 | AJ Green vs. Norman Powell | SCQ: 38.24 | Result: **Made**

![Frame 5724 — weak contest, Powell too slow and too far, shot converted](../data/outputs/visualizations/report/game 0022500062 | frame 5724.png)

Powell contests from 5.8 ft at a closing speed of only 2.3 ft/s — roughly a quarter of Mitchell's pace in Figure 3. Hand height is 37.0 in, which earns full formula credit but cannot compensate for the slow, distant approach. The image shows Powell's skeleton (blue) several feet to the side while Green (green) rises near the right baseline. The shot converted, consistent with a below-average process score. This is the clearest illustration of SCQ's purpose: process was poor (moderate distance, insufficient closing urgency), outcome followed. Contrast this with Figure 1, where a similarly poor process still produced a miss — underscoring why outcome-only metrics mislead and why the process score and the lift metric together tell a more complete story.
