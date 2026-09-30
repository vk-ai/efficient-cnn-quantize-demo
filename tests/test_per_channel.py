"""Round 4: per-channel weight quantization + per-layer error table (depthwise focus)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from efficient_cnn.config import load_config
from efficient_cnn.eval import run_before_after
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.qat import prepare_qat
from efficient_cnn.quantize import (
    QPARAM_NBYTES,
    WEIGHT_KEYS,
    _affine_params,
    fake_quantize_tensor,
    format_layer_error_table,
    layer_error_table,
    per_channel_params,
    quantize_model,
    quantize_weight,
    rescale_depthwise_channel,
)


def _outlier_dw(seed: int = 0) -> np.ndarray:
    """Depthwise-shaped (C, 1, 3, 3) weights with one wide-range channel."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=(8, 1, 3, 3))
    w[3] *= 40.0
    return w


def test_per_channel_params_one_per_output_channel():
    w = _outlier_dw()
    scales, zps = per_channel_params(w, bits=8)
    assert scales.shape == (8,) and zps.shape == (8,)
    wq, s_vec, _ = quantize_weight(w, 8, granularity="per_channel")
    np.testing.assert_allclose(s_vec, scales)
    # each channel's error is bounded by half of *its own* LSB
    err = np.abs(w - wq).reshape(8, -1).max(axis=1)
    assert np.all(err <= scales * 0.5 + 1e-12)


def test_outlier_channel_per_channel_beats_per_tensor():
    w = _outlier_dw()
    for bits in (8, 4):
        wt, _, _ = quantize_weight(w, bits, granularity="per_tensor")
        wc, _, _ = quantize_weight(w, bits, granularity="per_channel")
        mse_t = np.mean((w - wt) ** 2)
        mse_c = np.mean((w - wc) ** 2)
        assert mse_c < mse_t
        # the small (non-outlier) channels are where per-tensor hurts most
        small = [c for c in range(8) if c != 3]
        assert np.mean((w[small] - wc[small]) ** 2) < 0.05 * np.mean((w[small] - wt[small]) ** 2)


def test_per_tensor_default_is_unchanged():
    rng = np.random.default_rng(3)
    m = TinyEfficientCNN(1, 8, 4, rng)
    qm, stats = quantize_model(m, bits=8)  # default granularity
    assert stats.granularity == "per_tensor"
    for name in WEIGHT_KEYS:
        w = m.named_params()[name]
        s, z = _affine_params(w, 8)
        np.testing.assert_array_equal(qm.named_params()[name], fake_quantize_tensor(w, s, z, 8))
    assert stats.nbytes_qparams == len(WEIGHT_KEYS) * QPARAM_NBYTES


def test_quantize_model_per_channel_stats_and_skip():
    rng = np.random.default_rng(4)
    m = TinyEfficientCNN(1, 8, 4, rng)
    qt, st = quantize_model(m, bits=8, granularity="per_tensor")
    qc, sc = quantize_model(m, bits=8, granularity="per_channel", skip=["fc"])
    assert sc.granularity == "per_channel" and sc.skipped == ["fc_w"]
    assert set(sc.channel_scale) == {"stem_w", "dw_w", "pw_w"}
    for name, vec in sc.channel_scale.items():
        assert len(vec) == m.named_params()[name].shape[0]
    # extra scale vectors are counted separately from the int8 payload
    assert sc.nbytes_qparams == (8 + 8 + 8) * QPARAM_NBYTES
    assert sc.nbytes_int8_with_qparams == sc.nbytes_int8 + sc.nbytes_qparams
    np.testing.assert_array_equal(qc.fc_w, m.fc_w)  # skipped stays float
    with pytest.raises(ValueError, match="granularity"):
        quantize_model(m, granularity="per_group")


def test_rescale_depthwise_channel_is_function_preserving():
    rng = np.random.default_rng(5)
    m = TinyEfficientCNN(1, 8, 4, rng)
    x = rng.random((16, 1, 8, 8))
    om = rescale_depthwise_channel(m, channel=0, factor=64.0)
    np.testing.assert_allclose(om.forward(x), m.forward(x), rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError):
        rescale_depthwise_channel(m, factor=-2.0)
    base = {r["layer"]: r for r in layer_error_table(m, (4,), x=x)}
    stress = {r["layer"]: r for r in layer_error_table(om, (4,), x=x)}
    assert stress["dw_w"]["channel_range_spread"] > 10 * base["dw_w"]["channel_range_spread"]
    # per-tensor output error explodes, per-channel is untouched by the rescale
    assert stress["dw_w"]["logit_mse_per_tensor"] > 10 * base["dw_w"]["logit_mse_per_tensor"]
    assert stress["dw_w"]["logit_mse_per_channel"] == pytest.approx(
        base["dw_w"]["logit_mse_per_channel"], rel=1e-6
    )


def test_layer_error_table_rows_and_depthwise_highlight():
    rng = np.random.default_rng(6)
    m = TinyEfficientCNN(1, 8, 4, rng)
    rows = layer_error_table(m, (8, 4))
    assert len(rows) == 2 * len(WEIGHT_KEYS)
    assert [r["layer"] for r in rows if r["is_depthwise"]] == ["dw_w", "dw_w"]
    for r in rows:
        assert r["rel_mse_per_tensor"] > 0 and np.isfinite(r["ratio"]) and r["ratio"] > 0
    # Not guaranteed per tensor (rounding / zero-point effects), but in aggregate
    # finer scales reconstruct the weights better.
    for bits in (8, 4):
        sub = [r for r in rows if r["bits"] == bits]
        assert sum(r["mse_per_channel"] for r in sub) < sum(r["mse_per_tensor"] for r in sub)
    text = format_layer_error_table(rows)
    assert text.count("◀ depthwise") == 2


def test_qat_prepare_per_channel_snaps_to_channel_grid():
    rng = np.random.default_rng(7)
    m = TinyEfficientCNN(1, 8, 4, rng)
    prepared, _ = prepare_qat(m, bits=8, granularity="per_channel")
    wq, _, _ = quantize_weight(m.dw_w, 8, granularity="per_channel")
    np.testing.assert_allclose(prepared.dw_w, wq)


def test_eval_report_granularity_rows_and_outlier_stress():
    cfg = copy.deepcopy(load_config())
    cfg["distill"]["enabled"] = False
    cfg["qat"]["enabled"] = False
    cfg["quantize"]["granularity"] = "per_channel"
    rep = run_before_after(cfg)
    assert rep["quantized"]["granularity"] == "per_channel"
    assert set(rep["granularity_ablation"]) == {
        "int8_per_tensor", "int8_per_channel", "int4_per_tensor", "int4_per_channel",
    }
    assert rep["granularity_ablation"]["int8_per_channel"]["acc"] >= rep["baseline"]["acc"] - 0.05
    assert any(r["is_depthwise"] for r in rep["layer_error_table"])
    o = rep["dw_outlier"]
    assert o["float_acc"] == pytest.approx(rep["baseline"]["acc"])
    assert o["int4_per_channel"]["acc"] > o["int4_per_tensor"]["acc"]
    assert o["int8_per_channel"]["acc"] >= o["int8_per_tensor"]["acc"]
    with pytest.raises(ValueError, match="granularity"):
        bad = copy.deepcopy(cfg)
        bad["quantize"]["granularity"] = "nope"
        run_before_after(bad)
