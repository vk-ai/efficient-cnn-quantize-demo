"""Magnitude pruning for dense numpy weight tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efficient_cnn.model import TinyEfficientCNN

WEIGHT_KEYS = ("stem_w", "dw_w", "pw_w", "fc_w")


@dataclass
class PruneStats:
    sparsity: float
    total_weights: int
    nonzero_before: int
    nonzero_after: int
    pruned: int


def magnitude_prune_tensor(w: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the lowest-magnitude fraction of elements (global within tensor)."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    flat = w.ravel()
    n = flat.size
    k = int(n * sparsity)
    if k <= 0:
        return w.copy()
    idx = np.argpartition(np.abs(flat), k)[:k]
    out = flat.copy()
    out[idx] = 0.0
    return out.reshape(w.shape)


def prune_model(model: TinyEfficientCNN, sparsity: float) -> tuple[TinyEfficientCNN, PruneStats]:
    out = model.copy()
    params = out.named_params()
    total = 0
    nz_before = 0
    nz_after = 0
    for name in WEIGHT_KEYS:
        w = params[name]
        total += w.size
        nz_before += int(np.count_nonzero(w))
        pruned_w = magnitude_prune_tensor(w, sparsity)
        params[name] = pruned_w
        nz_after += int(np.count_nonzero(pruned_w))
    out.set_params(params)
    stats = PruneStats(
        sparsity=sparsity,
        total_weights=total,
        nonzero_before=nz_before,
        nonzero_after=nz_after,
        pruned=nz_before - nz_after,
    )
    return out, stats
