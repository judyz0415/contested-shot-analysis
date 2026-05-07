#!/usr/bin/env python3
"""
Create summary visualizations for defensive effectiveness outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_defender_lift(def_rows: List[Dict[str, str]], out_path: str) -> None:
    rows = []
    for r in def_rows:
        rows.append(
            {
                "name": r["nearest_defender_name"],
                "shots": int(r["shots"]),
                "lift": float(r["lift_vs_baseline_rate"]),
            }
        )
    rows.sort(key=lambda x: x["lift"])
    names = [f"{r['name']} ({r['shots']})" for r in rows]
    lifts = [r["lift"] for r in rows]
    colors = ["#1f77b4" if v < 0 else "#d62728" for v in lifts]

    plt.figure(figsize=(10, 6))
    y = range(len(rows))
    plt.barh(y, lifts, color=colors, alpha=0.85)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.yticks(y, names, fontsize=9)
    plt.xlabel("Lift vs baseline make rate (actual - expected)")
    plt.title("Defender Effectiveness (negative is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_calibration(
    cal_base: List[Dict[str, str]], cal_full: List[Dict[str, str]], out_path: str
) -> None:
    xb = [float(r["pred_mean"]) for r in cal_base]
    yb = [float(r["obs_rate"]) for r in cal_base]
    xf = [float(r["pred_mean"]) for r in cal_full]
    yf = [float(r["obs_rate"]) for r in cal_full]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
    plt.plot(xb, yb, marker="o", linewidth=2, label="Baseline")
    plt.plot(xf, yf, marker="o", linewidth=2, label="Full")
    plt.xlabel("Predicted make probability")
    plt.ylabel("Observed make rate")
    plt.title("Calibration: Baseline vs Full Model")
    plt.xlim(0.15, 0.9)
    plt.ylim(0.1, 0.9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_probability_shift(shot_rows: List[Dict[str, str]], out_path: str) -> None:
    p_base = [float(r["p_base"]) for r in shot_rows]
    p_full = [float(r["p_full"]) for r in shot_rows]

    plt.figure(figsize=(7, 6))
    plt.scatter(p_base, p_full, s=20, alpha=0.5)
    lo = min(min(p_base), min(p_full))
    hi = max(max(p_base), max(p_full))
    plt.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    plt.xlabel("Baseline predicted make probability")
    plt.ylabel("Full-model predicted make probability")
    plt.title("How Contest + Defender Terms Shift Predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize defensive effectiveness outputs.")
    parser.add_argument("--out-dir", default="data/outputs", help="Output directory with CSV artifacts.")
    args = parser.parse_args()

    defender_csv = os.path.join(args.out_dir, "defender_effectiveness_lift.csv")
    shot_csv = os.path.join(args.out_dir, "defensive_effectiveness_shot_level.csv")
    cal_base_csv = os.path.join(args.out_dir, "calibration_baseline.csv")
    cal_full_csv = os.path.join(args.out_dir, "calibration_full.csv")

    defender_rows = _read_csv(defender_csv)
    shot_rows = _read_csv(shot_csv)
    cal_base = _read_csv(cal_base_csv)
    cal_full = _read_csv(cal_full_csv)

    _plot_defender_lift(defender_rows, os.path.join(args.out_dir, "viz_defender_lift_vs_baseline.png"))
    _plot_calibration(cal_base, cal_full, os.path.join(args.out_dir, "viz_calibration_baseline_vs_full.png"))
    _plot_probability_shift(shot_rows, os.path.join(args.out_dir, "viz_probability_shift_base_vs_full.png"))

    print("Wrote visualizations:")
    print(" ", os.path.join(args.out_dir, "viz_defender_lift_vs_baseline.png"))
    print(" ", os.path.join(args.out_dir, "viz_calibration_baseline_vs_full.png"))
    print(" ", os.path.join(args.out_dir, "viz_probability_shift_base_vs_full.png"))


if __name__ == "__main__":
    main()
