# Defensive Effectiveness Summary

Baseline model uses shooter skill + shot context. Full model adds contest features and defender identity with ridge shrinkage. Defender ranking below uses volume-normalized lift so low-sample defenders are pulled toward neutral (0).

---

## Snapshot Metrics

- **Shots Modeled:** 399
- **Overall Make Rate:** 47.1%
- **Baseline Logloss:** 0.6818
- **Full Logloss:** 0.6568

> **Heat-facing interpretation**  
> For front office and coaching decisions, prioritize volume-normalized lift and shot volume together. This avoids overreacting to very small samples while still capturing true suppression signal.

---

## Volume-Normalized Defender Effectiveness

| Defender | Shots | Raw lift | Normalized lift | Takeaway |
|---|---:|---:|---:|---|
| Bam Adebayo | 44 | -0.120 | -0.077 | Most reliable suppression profile in sample |
| Nikola Jovic | 8 | -0.203 | -0.049 | Promising but low-volume; monitor, avoid overconfidence |
| Norman Powell | 22 | -0.091 | -0.043 | Moderate positive suppression signal |
| Pelle Larsson | 47 | +0.078 | +0.051 | Sustained over-baseline makes allowed |
| Jaime Jaquez Jr. | 46 | +0.095 | +0.062 | Needs targeted closeout/angle adjustment |
| Davion Mitchell | 57 | +0.109 | +0.076 | Largest adverse lift in this sample |

---

## SCQ Driver Breakdown (Heat Defenders)

Positive values indicate this component contributes more SCQ points than the Heat defender pool average; negative values indicate a drag on that defender's SCQ.

| Defender | Shots | Dist vs pool | Speed vs pool | Angle vs pool | Hand vs pool |
|---|---:|---:|---:|---:|---:|
| Davion Mitchell | 63 | +2.47 | +1.85 | -0.06 | -0.01 |
| Bam Adebayo | 55 | -2.45 | -3.01 | -0.14 | +0.84 |
| Jaime Jaquez Jr. | 53 | +0.46 | +1.12 | +0.18 | -0.02 |
| Tyler Herro | 49 | +0.41 | +0.77 | -0.03 | -1.72 |
| Pelle Larsson | 55 | -0.16 | -0.16 | +0.00 | -0.31 |
| Nikola Jovic | 10 | +2.03 | -0.40 | -0.14 | +1.99 |

---

## SCQ Coaching Recommendations

| Defender | Primary score drag | Action recommendation |
|---|---|---|
| Bam Adebayo | Speed and distance | Higher starting position + first-step acceleration closeout reps |
| Tyler Herro | Hand timing | Early high-hand at gather; drill no-dip hand timing into contests |
| Jaime Jaquez Jr. | Hand timing | Preserve speed but raise/earlier hand to convert pressure into SCQ |
| Pelle Larsson | Hand and speed | Pair hand-up timing cues with urgency benchmarks in shell closeouts |
| Nikola Jovic | Speed (low sample) | Keep distance/hand gains; focus on first two steps and angle control |
| Davion Mitchell | Angle | Maintain pressure profile while reducing side-angle/fly-by contests |

---

## Heat Recommendations

| Audience | Recommendation | Why it matters |
|---|---|---|
| Front Office | Use normalized lift + shots as the main defensive contest KPI. | Stabilizes evaluation across uneven workloads and avoids small-sample noise. |
| Coaching Staff | Prioritize film + drill work for high-volume positive-lift defenders. | Largest practical gain opportunity comes from frequent contest reps. |
| Players | For Jaquez/Larsson/Mitchell contests, emphasize angle and hand-up timing. | Model coefficients suggest better contest geometry lowers make odds. |
| Rotation Planning | Lean late-game high-leverage 3PT contests toward strongest normalized suppressors. | Improves expected opponent shot outcomes versus baseline shot quality. |

---

## Generated Visuals and Data Artifacts

- `data/outputs/viz_defender_lift_vs_baseline.png`
- `data/outputs/viz_defender_lift_volume_normalized.png`
- `data/outputs/viz_calibration_baseline_vs_full.png`
- `data/outputs/viz_probability_shift_base_vs_full.png`
- `data/outputs/defender_scq_driver_breakdown.csv`
- `data/outputs/defender_scq_recommendations.csv`

Calibration CSVs and shot-level lift outputs remain in `data/outputs/` for downstream modeling and reporting.
