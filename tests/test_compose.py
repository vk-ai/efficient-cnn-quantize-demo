"""Tests for ordered prune → int8 compose path."""

from __future__ import annotations

import numpy as np

from efficient_cnn.config import load_config
from efficient_cnn.eval import apply_compose, run_before_after
from efficient_cnn.metrics import param_count
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.prune import prune_model
from efficient_cnn.quantize import quantize_model


def test_compose_order_from_config():
    cfg = load_config()
    assert cfg["compose"]["order"] == ["prune", "quantize"]


def test_apply_compose_prune_then_quantize_matches_manual():
    rng = np.random.default_rng(11)
    model = TinyEfficientCNN(1, 8, 4, rng)
    composed, meta = apply_compose(model, ["prune", "quantize"], bits=8, sparsity=0.5)
    assert meta["order"] == ["prune", "quantize"]
    assert "prune" in meta["steps"] and "quantize" in meta["steps"]

    manual, _ = prune_model(model, 0.5)
    manual, _ = quantize_model(manual, bits=8)
    for key in ("stem_w", "dw_w", "pw_w", "fc_w"):
        assert np.allclose(
            composed.named_params()[key],
            manual.named_params()[key],
        )


def test_eval_includes_compose_row_vs_baselines():
    cfg = load_config()
    report = run_before_after(cfg)
    assert "prune_then_int8" in report
    row = report["prune_then_int8"]
    assert row["order"] == ["prune", "quantize"]
    # Compose should be at least as sparse as prune-only (same prune, then fake-quant).
    assert row["nonzero"] <= report["pruned"]["nonzero"]
    # Packed int8 size should beat float baseline payload.
    assert row["nbytes_int8_pack"] < report["baseline"]["nbytes_fp64"]
    # Accuracy stays in a loose teaching band on the synthetic blobs.
    assert row["acc"] >= report["baseline"]["acc"] - 0.45
    assert report["compose"]["order"] == ["prune", "quantize"]
