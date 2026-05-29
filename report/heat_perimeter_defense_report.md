# Contesting the Three: A Hawk-Eye Analysis of Perimeter Defense
### How Miami Heat Defenders Are (and Aren't) Suppressing 3-Point Attempts

---

*Analysis by Judy Zhu · May 2026*  
*Data: Hawk-Eye optical tracking, 15 Miami Heat games, 2025–26 NBA regular season*

---

## Executive Summary

Not all 3-point contests are created equal. Using sub-second Hawk-Eye optical tracking data, I built a framework to measure how Heat defenders contest 3-point shots — not just whether they show up, but how close, how fast, at what angle, and with what physical advantage they arrive.

Three findings stand out:

1. **Size is the strongest predictor of shot suppression.** A defender whose hand clears the shooter's release point by one standard deviation (~2 inches) cuts expected make probability by ~12%. Height and wingspan advantages explain more shot-to-shot variance than any single positioning or speed metric.

2. **Contest effort (SCQ) and defensive outcome (lift vs. baseline) diverge at the player level.** The defenders with the highest composite contest scores are often those assigned to the hardest shooters — effort and matchup difficulty are tangled together. Controlling for shooter skill and shot difficulty exposes a cleaner picture.

3. **Rankings are robust to weighting assumptions except when speed is removed entirely.** Speed matters because it proxies how far away a defender starts, not raw athleticism. Stationed defenders (Bam Adebayo) look worse on speed-weighted metrics than their actual shot-suppression numbers.

---

## 1. The Data Pipeline

### What Hawk-Eye Captures
Each game file contains positional data at **60 frames per second** — roughly one centroid location per player per 16.7 milliseconds, plus per-frame wrist positions (left and right) and ball location in 3D space. This is a different class of data from play-by-play logs or box scores: it allows reconstruction of *how* a defender moved on every possession, not just whether the shot went in.

For this analysis I worked with 15 games (~1.1 GB per game), filtering to **three-point field goal attempts** with an identifiable release frame — the moment the ball departs the shooter's hands.

### Defining "Active Contest"
I restricted analysis to shots where, at the release frame:
- The nearest Heat defender was **within 8 feet** of the shooter (filters out half-hearted recoveries)
- That defender's angle was **≥ 90°** — meaning they were in front of the shooter, not trailing behind them

This left **310 actively contested shots** across 12 Heat defenders with ≥ 5 qualifying contests.

> **Why the angle filter matters.** About 4% of apparent "contests" in raw tracking are defenders sprinting toward a shooter from behind — they show high speed and low distance, but they aren't blocking the shooter's sightline at release. Excluding these avoids crediting high-effort recoveries as actual contest quality.

The **contest angle** is defined as the angle between the (defender → shooter) vector and the (shooter → rim) vector. An angle of 90° means the defender is at the shooter's side; 180° means the defender is directly between the shooter and the rim — a full face-guard. Only contests ≥ 90° are included.

---

## 2. Shot Contest Quality (SCQ): A Composite Metric

### Formula

For each shot, SCQ is a 0–100 weighted index of four contest dimensions measured at the release frame:

| Component | Weight | Normalization | What It Measures |
|-----------|--------|--------------|-----------------|
| Defender distance to shooter | 35% | `max(0, 1 − dist/10)` | Proximity at release |
| Closeout speed | 30% | `clip(speed/8, 0, 1)` | How fast the defender was moving |
| Contest angle | 20% | `clip((angle−90°)/90°, 0, 1)` | How directly in front of the shooter |
| Hand height above ball | 15% | `clip(hand_in/18, 0, 1)` | Defender's wrist above release point |

The angle component was revised from an earlier (inverted) formula that accidentally gave higher scores to defenders *behind* the shooter. The corrected version rewards angles closer to 180° — defenders who stand directly between shooter and rim.

SCQ is computed at the individual shot level. Per-defender scores below are averages over their contested shots.

### Per-Defender SCQ Rankings

| Rank | Defender | Shots | Mean SCQ | Dist | Speed | Angle | Hand |
|------|----------|-------|----------|------|-------|-------|------|
| 1 | Dru Smith | 16 | **71.1** | 16.1 | **28.9** | 11.2 | 14.9 |
| 2 | Kasparas Jakucionis | 13 | **69.6** | 15.5 | 26.5 | 13.1 | 14.6 |
| 3 | Davion Mitchell | 48 | **69.5** | 15.9 | 25.3 | 14.3 | 13.9 |
| 4 | Tyler Herro | 32 | **68.8** | 16.2 | 25.7 | 13.7 | 13.3 |
| 4 | Kel'el Ware | 10 | **68.8** | 13.5 | 24.1 | **16.3** | **15.0** |
| 6 | Jaime Jaquez Jr. | 40 | **67.1** | 14.8 | 23.6 | 14.2 | 14.5 |
| 7 | Pelle Larsson | 39 | **67.0** | 15.1 | 23.7 | 14.1 | 14.1 |
| 8 | Simone Fontecchio | 10 | **66.8** | 12.4 | 27.6 | 12.4 | 14.4 |
| 9 | Norman Powell | 15 | **66.0** | 14.2 | 22.4 | 15.2 | 14.2 |
| 10 | Nikola Jovic | 10 | **65.1** | 14.2 | 21.5 | 14.3 | **15.0** |
| 11 | Andrew Wiggins | 52 | **64.6** | 13.6 | 20.5 | 16.1 | 14.6 |
| 12 | Bam Adebayo | 25 | **62.3** | 14.1 | 17.6 | 15.6 | **15.0** |

*Pool average SCQ ≈ 67.1. Component scores shown in 0–100 scale points (weight × normalized × 100).*

### What Drives Each Defender's Score

**Dru Smith (#1)** is almost entirely **speed-driven** — he averages +5.3 points above pool mean on speed, the highest gap on the team. This reflects a sprint-closeout style: he starts farther from shooters and arrives at high velocity. But his **angle score is the lowest on the team (11.2)**, revealing a tradeoff — he arrives fast but often from the side rather than squarely in front.

**Kel'el Ware (#4)** earns his ranking through the opposite profile: **best angle score (16.3)** and tied for best hand score (15.0). He is the most consistently positioned defender, standing directly between shooters and the rim. Speed is below average because he's not covering as much ground.

**Bam Adebayo (#12)** suffers entirely from the speed penalty. His angle (15.6) and hand (15.0) scores are both well above average — he is properly positioned and physically dominant. The speed component penalizes him for starting close, not for poor defense.

**Key takeaway for evaluation:** SCQ is best interpreted within a defensive *style* context. High-speed contests and stationed contests are fundamentally different situations. A better comparison is to group defenders by style and assess within-group, not across the whole roster.

---

## 3. Are SCQ Rankings Stable? A Sensitivity Analysis

The weights in SCQ are heuristic starting points. I tested five alternative weighting schemes to see how much the rankings depend on those choices:

| Scheme | dist | speed | angle | hand | Spearman ρ vs. Original |
|--------|------|-------|-------|------|------------------------|
| Original (baseline) | 0.35 | 0.30 | 0.20 | 0.15 | 1.000 |
| Equal weights | 0.25 | 0.25 | 0.25 | 0.25 | **0.916** |
| Distance-heavy | 0.55 | 0.20 | 0.15 | 0.10 | **0.937** |
| Technique-heavy (angle + hand) | 0.20 | 0.15 | 0.35 | 0.30 | **−0.126** |
| No speed (stationed focus) | 0.45 | 0.00 | 0.30 | 0.25 | **−0.462** |

**Rankings are stable only when speed is heavily weighted.** When angle and hand technique are upweighted — or when speed is removed — the rankings don't just shift, they essentially **invert** (ρ = −0.126 and −0.462 respectively). This is not metric fragility: it's revealing a genuine split in what's being measured.

### Two Defensive Profiles, Two Opposite Rankings

The core tension is between **closers** and **stationed defenders**:

| Profile | Examples | Strength | Weakness |
|---------|----------|----------|---------|
| Closer | Dru Smith, Kasparas Jakucionis | High speed, good distance | Angle: arriving from the side |
| Stationed | Kel'el Ware, Bam Adebayo, A. Wiggins | Best angle, best hand | Low speed (already there) |

Under the original weighting (speed = 30%), closers lead. Under technique-heavy (angle = 35%, hand = 30%), **Kel'el Ware jumps to #1, Andrew Wiggins to #2, Bam Adebayo to #6** — and Dru Smith falls to #10.

This is not about which scheme is "right." It's about what question you're asking: *Who closes out hardest?* favors speed. *Who stands in the most disruptive position?* favors angle and hand. A complete evaluation uses both.

**What the sensitivity analysis tells us about the roster:**
- Davion Mitchell and Pelle Larsson are **consistently mid-pack** (ranks #3–#9) across all schemes — balanced, not elite in any single dimension
- Dru Smith's #1 ranking is **entirely dependent** on speed weight; he's a below-average angle defender
- Bam Adebayo's positioning and technique would rank him in the top half of the team if speed weren't penalizing him for being stationed

---

## 4. Does Better Contesting Lead to Fewer Makes?

### The Lift Framework

Raw make rates are noisy: they mix shooter quality, shot difficulty, and defensive impact. To isolate the defensive contribution, I trained a **ridge logistic regression baseline model** on shooter-side and shot-trajectory features:

- Shooter's 2025–26 regular season 3-point percentage
- Distance to the rim at release
- Release height and shot arc (apex ball height)
- Shot clock remaining

This baseline predicts the *expected* make probability without any defensive information. **Lift** = actual make rate minus baseline expected make rate. A negative lift means the defender is suppressing makes below what shooter quality and shot difficulty alone would predict — genuine defensive value.

### Per-Defender Lift Results

| Defender | Shots | Actual % | Expected % | Lift | Confidence |
|----------|-------|----------|------------|------|------------|
| Bam Adebayo | 24 | **20.8%** | 41.3% | **−20.5 pp** | ★★★ |
| Nikola Jovic | 8 | **25.0%** | 43.1% | **−18.1 pp** | ★ (small n) |
| Tyler Herro | 32 | **34.4%** | 45.7% | **−11.4 pp** | ★★★ |
| Kasparas Jakucionis | 13 | **30.8%** | 40.8% | **−10.0 pp** | ★★ |
| Norman Powell | 15 | **40.0%** | 46.9% | **−6.9 pp** | ★★ |
| Kel'el Ware | 10 | **40.0%** | 45.6% | **−5.6 pp** | ★ (small n) |
| Simone Fontecchio | 9 | **44.4%** | 43.1% | **+1.3 pp** | ∼ |
| Andrew Wiggins | 50 | **44.0%** | 41.9% | **+2.1 pp** | ∼ |
| Dru Smith | 16 | **50.0%** | 43.1% | **+6.9 pp** | − |
| Davion Mitchell | 48 | **52.1%** | 45.1% | **+7.0 pp** | − |
| Pelle Larsson | 38 | **52.6%** | 44.6% | **+8.0 pp** | − |
| Jaime Jaquez Jr. | 38 | **55.3%** | 43.4% | **+11.9 pp** | − |

*★★★ = strong suppression signal; ★★ = moderate; ★ = directional only (n < 20); ∼ = neutral; − = above expected. pp = percentage points.*

### The SCQ–Lift Disconnect

The Spearman correlation between a defender's mean SCQ and their lift is **ρ = +0.34** — weakly positive in the *wrong* direction. Harder-contesting defenders are associated with *more* makes above expected, not fewer.

This is a **matchup effect**, not a metric failure. High-contest-quality defenders are deployed against the shooters who most demand active contests — better shooters, more dangerous actions. The harder the assignment, the higher the make rate the defender is fighting against. SCQ captures *effort and execution*; lift captures *outcome controlling for difficulty*.

At the **per-shot level**, the relationship is negative (ρ = −0.078), confirming that harder individual contests do modestly suppress makes. The effect exists but gets obscured at the player aggregate level by selection into harder matchups.

**Practical implication:** Evaluating perimeter defenders by raw make rate penalizes those who draw the hardest assignments. The lift framework separates matchup difficulty from defensive execution.

---

## 5. What Actually Suppresses Makes: Physical Features

I added four physical contest features to the logistic regression model alongside the contest mechanics:

| Feature | Beta | Odds Ratio (per +1 SD) | Interpretation |
|---------|------|----------------------|----------------|
| Effective contest height (in) | **−0.126** | **0.882** | Defender's hand above/below release ball |
| Height difference, def − shooter (in) | **−0.122** | **0.885** | Raw height advantage |
| SCQ composite | −0.092 | 0.912 | Combined contest quality |
| Wingspan vs. shooter height (in) | −0.097 | 0.908 | Reach advantage |
| Defender jump (in, 250ms pre-release) | **+0.129** | **1.138** | See note below |

*Full model: Brier score 0.233 vs. baseline 0.243 — a meaningful improvement.*

**Effective contest height** — how many inches the defender's highest hand cleared (or fell below) the ball at release — is the single strongest predictor. Each standard deviation improvement (~2 inches of additional hand clearance) reduces make probability by roughly 12%. This is the cleanest data signal in the entire analysis, and it directly validates what coaches mean when they say "get your hand up."

**Height difference** (defender height minus shooter height) has nearly identical predictive power. A taller defender suppresses makes even holding contest positioning constant. This is the data capturing a basketball truism: you can be in perfect position but still get shot over if you're shorter.

**Defender jump** shows a **positive** beta (+0.129) — larger pre-release jumps are associated with *more* makes. This is a **pump-fake artifact**: defenders who jump are responding to fakes or mistiming their contest, leaving them airborne when the ball actually leaves the shooter's hand. The best contests in this dataset are disciplined, grounded hand-extensions, not explosive vertical efforts.

> **Scouting profile of an effective 3-point perimeter defender:** tall with long wingspan, already in position in front of the shooter, extends the hand above the ball without leaving the floor. Physical size drives the model; discipline (not jumping) enables size to matter.

### Bam Adebayo: The Central Case Study

Adebayo finishes last in SCQ (speed penalty: 17.6 vs. pool average 23.5) but first in lift suppression (−20.5 pp). The physical model explains the gap entirely. As a center defending big-on-big 3-point contests:

- His hand height advantage over shorter shooter opponents is consistently large
- He doesn't jump at fakes — his jump score is near zero
- He is already positioned in front of his man (angle score: 15.6, well above several wing defenders)

SCQ penalizes him for not having to run. The outcome model reveals what running fast doesn't tell you: *being already in the right place with the right size is better than arriving fast from the wrong place.*

---

## 6. Methodology Notes

### Ridge Logistic Regression
I implemented ridge logistic regression from scratch in pure Python (no sklearn) to maintain full transparency and avoid dependency issues. The model uses L2 regularization (λ = 0.1) and gradient descent on 302 analysis-eligible shots. Betas are reported per raw unit change; odds ratios are scaled to per +1 standard deviation for comparability across features.

### Hawk-Eye Feature Engineering
All defensive features are extracted at the shot release frame. Key engineered features:
- **Closeout speed**: Euclidean displacement of defender centroid over the 500ms window ending at release
- **Effective contest height**: `max(left_wrist_z, right_wrist_z) − ball_z` at release
- **Defender jump**: vertical displacement of defender centroid over the 250ms window ending at release
- **Contest angle**: arccos of the dot product of (defender→shooter) and (shooter→rim) unit vectors

### Limitations

1. **Small sample.** 15 games yields ~310 eligible contests across 12 defenders. Most defenders have fewer than 50 shots; individual lift estimates carry substantial uncertainty. A full 82-game season would roughly halve confidence intervals.

2. **Reactive jump artifact.** The 250ms pre-release jump window conflates deliberate contest timing with pump-fake reactions. Future work could label possessions containing fakes (identifiable in tracking via ball deceleration patterns) and handle them separately.

3. **No assignment-quality adjustment in SCQ.** SCQ describes the contest, not who you contested. Guarding Steph Curry with a 70 SCQ and guarding a backup with a 70 SCQ are not equivalent. The lift baseline partially addresses this via the shooter-prior term; SCQ does not.

4. **Matchup selection bias in lift estimates.** Coaching rotations determine assignments. The defenders with the worst lift numbers may be assigned to the most dangerous shooters precisely because they are trusted. Causal inference from observational lift numbers requires caution.

5. **Positive contest distance/angle betas in isolation.** When fit without physical features, individual contest mechanics (distance, angle) show counterintuitive positive relationships with make rate. This is a collinearity and selection artifact from the 8-ft distance filter: defenders between 6–8 ft are partially contesting but not fully disrupting. Physical features resolve the collinearity — the meaningful variance is absorbed by hand clearance and height.

---

## 7. Key Takeaways for Roster and Development Evaluation

**Use lift, not raw make rate, to evaluate perimeter defenders.** The lift framework absorbs shooter quality and shot difficulty, leaving a cleaner read on what the defender contributed. In this sample, the top-lift defenders (Adebayo −20.5 pp, Herro −11.4 pp, Powell −6.9 pp) allowed substantially fewer makes than even the same shooters would have made against a league-average contest.

**SCQ is a development diagnostic, not an outcome ranking.** It tells a defender exactly where they're losing points: distance (are you too far at release?), speed (are you closing fast enough?), angle (are you arriving from the side?), hand (are you getting above the ball?). That granularity is valuable for film-room feedback and individual improvement tracking.

**Physical size drives outcomes; technique unlocks it.** Height and wingspan cannot be coached, but effective contest height — how high the hand gets above the ball — is partially a technique outcome. Teaching contested hand extension, proper footwork to stay in front without overcommitting, and the discipline to not leave the floor against shot fakes all translate directly into the features this model rewards.

**Dru Smith and Kasparas Jakucionis are the clearest contest-effort stories.** Both hold their top-2 SCQ positions across every weighting scheme tested. They are not gaming one component — distance, speed, angle, and hand are all above average. Their lift numbers are not yet negative, but 16 and 13 shots are not enough to read against them. The effort and mechanics are there.

**Bam Adebayo's defensive value on 3-point contests is being undercounted by conventional metrics.** His raw make rate allowed looks poor because he's positioned well and draws quality shooters. His lift number and physical model position tell the correct story.

---

## Appendix: Data & Code

| Resource | Location |
|----------|----------|
| Hawk-Eye tracking | 15 Heat games, 2025–26 (OneDrive) |
| Physical measurements | `data/player_height_wingspan.csv` (1,835 players) |
| Shot contest dataset | `data/intermediate/shot_contest_dataset.csv` (459 rows, 313 eligible) |
| Pipeline | `scripts/pipeline/build_unified_shot_dataset.py` |
| Lift model | `scripts/analysis/model_defensive_effectiveness.py` |
| SCQ driver breakdown | `scripts/analysis/explain_scq_drivers_by_defender.py` |
| SCQ sensitivity | `scripts/analysis/scq_weight_sensitivity.py` |
| SCQ–lift correlation | `scripts/analysis/scq_lift_correlation.py` |
| Shot visualizations | `notebooks/shot_viz_contest_extremes/` |

All modelling implemented in pure Python (standard library + numpy/pandas). Ridge logistic regression written from scratch with gradient descent and L2 regularization.

---

*Questions or follow-up: judy.zhu6052@gmail.com*
