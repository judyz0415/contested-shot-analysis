# Defensive Effectiveness Summary

SCQ (Shot Contest Quality) summarizes Hawk-Eye contest geometry at release into a 0–100 score. The tables below use **analysis-eligible** opponent three-point attempts only (**509** contests across **15** games). **Pool deltas** compare each defender’s average implied SCQ points per driver (distance, speed, angle, hand) to the mean over all **509** eligible rows.

**Deliverable:** merged breakdown + coaching strings → `data/outputs/heat_defender_scq_breakdown_final.csv`  
**Threshold:** nearest Heat defender with **≥5** eligible contested 3s → **12** players.

---

## Snapshot

| Metric | Value |
|--------|------:|
| Games | 15 |
| Eligible contests (pool) | 509 |
| Heat defenders (≥5 contests) | 12 |
| Pool mean SCQ | 47.93 |

---

## SCQ driver breakdown (mean SCQ and Δ vs pool)

Positive Δ = more contribution from that driver than the pool average (in SCQ points on the same 0–100 construction).

| Defender | Shots | Mean SCQ | Dist Δ | Speed Δ | Angle Δ | Hand Δ |
|----------|------:|---------:|-------:|--------:|--------:|-------:|
| Andrew Wiggins | 88 | 47.05 | −0.38 | −0.97 | +0.03 | +0.43 |
| Davion Mitchell | 72 | 51.61 | +1.78 | +2.21 | −0.12 | −0.19 |
| Pelle Larsson | 61 | 48.80 | +0.38 | +0.36 | −0.09 | +0.22 |
| Bam Adebayo | 57 | 42.15 | −2.81 | −3.37 | −0.20 | +0.60 |
| Jaime Jaquez Jr. | 57 | 49.03 | +0.75 | +0.36 | +0.11 | −0.12 |
| Tyler Herro | 47 | 46.45 | +0.23 | +0.26 | −0.08 | −1.89 |
| Dru Smith | 26 | 56.10 | +3.19 | +3.66 | +0.55 | +0.76 |
| Norman Powell | 25 | 43.14 | −1.29 | −2.18 | −0.20 | −1.12 |
| Kel'el Ware | 20 | 43.93 | −3.41 | −1.30 | −0.20 | +0.91 |
| Kasparas Jakucionis | 20 | 48.68 | +0.69 | −0.63 | +0.27 | +0.43 |
| Simone Fontecchio | 18 | 48.51 | −1.35 | +3.55 | −0.20 | −1.42 |
| Nikola Jovic | 13 | 49.44 | +0.91 | −2.03 | +1.05 | +1.59 |

---

## Leverage vs pool (strongest / weakest SCQ drivers)

| Defender | Strongest vs pool | Weakest vs pool |
|----------|-------------------|-----------------|
| Andrew Wiggins | hand | speed |
| Davion Mitchell | speed | hand |
| Pelle Larsson | distance | angle |
| Bam Adebayo | hand | speed |
| Jaime Jaquez Jr. | distance | hand |
| Tyler Herro | speed | hand |
| Dru Smith | speed | angle |
| Norman Powell | angle | speed |
| Kel'el Ware | hand | distance |
| Kasparas Jakucionis | distance | speed |
| Simone Fontecchio | speed | hand |
| Nikola Jovic | hand | speed |

Full drill and film cues (same logic as the script’s recommendation helper) are in **`heat_defender_scq_breakdown_final.csv`** columns `coaching_drill_recommendation` and `film_focus_recommendation`.

---

## Outcome modeling (separate path)

Residual **lift vs shooter-context baseline** (ridge logit) is produced by `scripts/analysis/model_defensive_effectiveness.py` when the shot CSV includes the baseline feature columns that script expects. Regenerate artifacts under `data/outputs/` after updating the unified dataset.

---

## Generated artifacts

- **`data/outputs/heat_defender_scq_breakdown_final.csv`** — primary SCQ export (means, components, pool diffs, coaching strings).
- `data/outputs/defender_scq_driver_breakdown.csv`
- `data/outputs/defender_scq_recommendations.csv`
- `data/outputs/viz_defender_lift_vs_baseline.png` (when the effectiveness model is run)
- `data/outputs/viz_defender_lift_volume_normalized.png`

Regenerate SCQ tables:

```bash
python scripts/analysis/explain_scq_drivers_by_defender.py \
  --csv data/intermediate/shot_contest_dataset.csv \
  --out-dir data/outputs \
  --min-shots 5 \
  --analysis-eligible-only \
  --final-csv data/outputs/heat_defender_scq_breakdown_final.csv
```
