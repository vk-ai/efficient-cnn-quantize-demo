"""Calibration observers + selective skip-list quantize."""

from __future__ import annotations

import numpy as np

from efficient_cnn.config import load_config
from efficient_cnn.eval import run_before_after
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.quantize import quantize_model


def test_percentile_observer_differs_from_minmax_on_heavy_tail():
    rng = np.random.default_rng(7)
    m = TinyEfficientCNN(1, 8, 4, rng)
    # Inject an outlier into stem weights so percentile clips differently
    params = m.named_params()
    params["stem_w"] = params["stem_w"].copy()
    params["stem_w"].flat[0] = 50.0
    m.set_params(params)

    q_mm, s_mm = quantize_model(m, bits=8, observer="minmax")
    q_pc, s_pc = quantize_model(m, bits=8, observer="percentile", percentile=99.0)
    assert s_mm.observer == "minmax"
    assert s_pc.observer == "percentile"
    # Scales for stem should differ when an outlier dominates minmax
    assert not np.isclose(s_mm.scale["stem_w"], s_pc.scale["stem_w"])
    assert not np.allclose(q_mm.named_params()["stem_w"], q_pc.named_params()["stem_w"])


def test_skip_fc_leaves_classifier_float():
    rng = np.random.default_rng(3)
    m = TinyEfficientCNN(1, 8, 4, rng)
    fc_before = m.named_params()["fc_w"].copy()
    q, stats = quantize_model(m, bits=8, observer="minmax", skip=["fc"])
    assert "fc_w" in stats.skipped
    assert "fc_w" not in stats.quantized
    assert np.allclose(q.named_params()["fc_w"], fc_before)
    # Other weights should still be quantized (not identical to float original in general)
    assert "stem_w" in stats.quantized
    # Pack size larger than full int8 (fc stays fp64)
    q_full, full_stats = quantize_model(m, bits=8, skip=[])
    assert stats.nbytes_int8 > full_stats.nbytes_int8


def test_eval_includes_calib_and_skip_rows():
    cfg = load_config()
    cfg["train"]["epochs"] = 2
    cfg["data"]["n_train"] = 48
    cfg["data"]["n_test"] = 24
    report = run_before_after(cfg)
    for key in ("int8_minmax", "int8_percentile", "int8_skip_head", "prune_then_int8"):
        assert key in report
        assert "acc" in report[key]
        assert "nbytes_int8_pack" in report[key] or key == "prune_then_int8"
    assert report["int8_minmax"]["observer"] == "minmax"
    assert report["int8_percentile"]["observer"] == "percentile"
    assert report["int8_skip_head"]["skipped"]
    # skip-head pack should be larger than full minmax pack
    assert report["int8_skip_head"]["nbytes_int8_pack"] > report["int8_minmax"]["nbytes_int8_pack"]
