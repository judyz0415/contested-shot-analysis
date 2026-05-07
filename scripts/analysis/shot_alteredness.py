# -*- coding: utf-8 -*-
"""
Pre-release ball path vs. release-frame position: 3D residual (inches) and scaled alteredness.

Linear model: for each coordinate, fit y(k) = slope * k + intercept to samples k = 0..n-1
corresponding to the n frames immediately before release; predict at virtual index k = n
(the release frame). Residual is Euclidean distance between predicted and observed release
position in Hawk-Eye inches.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def _predict_next_linear(ys: Sequence[float], n: int) -> float:
    """ys is length n with time indices k = 0..n-1; return prediction at index n."""
    if n < 3 or len(ys) != n:
        return float("nan")
    if any((not math.isfinite(v)) for v in ys):
        return float("nan")
    sum_k = n * (n - 1) / 2.0
    sum_kk = (n - 1) * n * (2 * n - 1) / 6.0
    sum_y = float(sum(ys))
    sum_ky = float(sum(k * ys[k] for k in range(n)))
    denom = n * sum_kk - sum_k * sum_k
    if abs(denom) < 1e-9:
        return float("nan")
    slope = (n * sum_ky - sum_k * sum_y) / denom
    intercept = (sum_y - slope * sum_k) / n
    return float(slope * n + intercept)


def path_residual_and_alteredness(
    ball_x: Sequence[float],
    ball_y: Sequence[float],
    ball_z: Sequence[float],
    release_index: int,
    window_frames: int = 12,
    alteredness_reference_inches: float = 12.0,
) -> Tuple[float, float]:
    """
    Return (release_path_residual_3d_inches, alteredness_on_0_100_scale).

    `release_index` is the row index in the (time-ordered) ball arrays where release is detected.
    The pre-release window uses indices [release_index - window_frames, release_index).
    """
    w = window_frames
    if w < 3:
        return float("nan"), float("nan")
    start = release_index - w
    if start < 0 or release_index >= len(ball_x):
        return float("nan"), float("nan")
    if len(ball_x) != len(ball_y) or len(ball_x) != len(ball_z):
        return float("nan"), float("nan")

    xs = [float(ball_x[i]) for i in range(start, release_index)]
    ys = [float(ball_y[i]) for i in range(start, release_index)]
    zs = [float(ball_z[i]) for i in range(start, release_index)]
    px = _predict_next_linear(xs, w)
    py = _predict_next_linear(ys, w)
    pz = _predict_next_linear(zs, w)
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
        return float("nan"), float("nan")
    ax = float(ball_x[release_index])
    ay = float(ball_y[release_index])
    az = float(ball_z[release_index])
    if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az)):
        return float("nan"), float("nan")
    residual = math.sqrt((ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2)
    if not math.isfinite(residual):
        return float("nan"), float("nan")
    if alteredness_reference_inches <= 0.0:
        return residual, float("nan")
    alteredness = round(100.0 * max(0.0, min(1.0, residual / alteredness_reference_inches)), 2)
    return residual, float(alteredness)
