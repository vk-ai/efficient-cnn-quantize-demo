"""Param counts and rough FLOPs for the tiny efficient CNN."""

from __future__ import annotations

import numpy as np

from efficient_cnn.model import TinyEfficientCNN


def param_count(model: TinyEfficientCNN, nonzero_only: bool = False) -> int:
    total = 0
    for arr in model.named_params().values():
        if nonzero_only:
            total += int(np.count_nonzero(arr))
        else:
            total += int(arr.size)
    return total


def estimate_flops(model: TinyEfficientCNN, height: int, width: int) -> int:
    """Rough multiply-add count for one forward (NCHW, N=1), ignoring ReLU/GAP."""
    c_in = model.in_channels
    w = model.width
    # stem 3x3: out_h≈h, out_w≈w after pad
    stem = w * c_in * 9 * height * width
    # depthwise 3x3
    dw = w * 9 * height * width
    # pointwise 1x1
    pw = w * w * 1 * height * width
    # fc
    fc = model.n_classes * w
    return int(stem + dw + pw + fc)
