"""Round 5: sensitivity-driven auto skip list / mixed precision under an accuracy budget."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from efficient_cnn.config import load_config
from efficient_cnn.eval import build_trained_model, run_auto_mixed_precision_demo, run_before_after
from efficient_cnn.mixed_precision import (
    auto_mixed_precision,
    format_mixed_precision,
    layer_nbytes,
    measure_sensitivity,
    model_nbytes,
    normalize_ladder,
    pareto_front,
    quantize_mixed,
)
from efficient_cnn.quantize import (
    QPARAM_NBYTES,
    WEIGHT_KEYS,
    layer_error_table,
    quantize_model,
    rescale_depthwise_channel,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained():
    cfg = load_config()
    train_ds, test_ds, model = build_trained_model(cfg)
    return cfg, train_ds, test_ds, model


@pytest.fixture(scope="module")
def demo(trained):
    cfg, train_ds, test_ds, model = trained
    return run_auto_mixed_precision_demo(cfg, model, train_ds, test_ds)


def test_normalize_ladder():
    assert normalize_ladder(["float", 8, 4, 2]) == ["float", 8, 4, 2]
    assert normalize_ladder(["fp64", "int8", "4"]) == ["float", 8, 4]
    for bad in ([8, 4], ["float"], ["float", 4, 8], ["float", 8, 8], ["float", 1], ["float", "x"]):
        with pytest.raises(ValueError):
            normalize_ladder(bad)


def test_quantize_mixed_matches_uniform_quantize_model(trained):
    _, _, test_ds, model = trained
    for bits in (8, 4):
        uni, _ = quantize_model(model, bits=bits)
        mixed = quantize_mixed(model, {k: bits for k in WEIGHT_KEYS})
        assert np.allclose(uni.forward(test_ds.X), mixed.forward(test_ds.X))
    same = quantize_mixed(model, {})
    assert np.array_equal(same.forward(test_ds.X), model.forward(test_ds.X))
    with pytest.raises(KeyError):
        quantize_mixed(model, {"nope_w": 8})


def test_nbytes_matches_quantize_model_accounting(trained):
    _, _, _, model = trained
    _, stats = quantize_model(model, bits=8)
    assert model_nbytes(model, {k: 8 for k in WEIGHT_KEYS}, "per_tensor") == (
        stats.nbytes_int8_with_qparams
    )
    assert model_nbytes(model, {}, "per_tensor") == stats.nbytes_fp64
    w = model.named_params()["dw_w"]
    assert layer_nbytes(model, "dw_w", 4, "per_tensor") == w.size / 2 + QPARAM_NBYTES
    assert layer_nbytes(model, "dw_w", 2, "per_channel") == w.size / 4 + w.shape[0] * QPARAM_NBYTES


def test_sensitivity_logit_mse_matches_round4_layer_error_table(trained):
    _, train_ds, _, model = trained
    x, y = train_ds.X[:64], train_ds.y[:64]
    sens = {(r["layer"], r["bits"]): r for r in measure_sensitivity(model, x, y, ["float", 8, 4])}
    for row in layer_error_table(model, (8, 4), x=x):
        assert np.isclose(sens[(row["layer"], row["bits"])]["logit_mse"],
                          row["logit_mse_per_tensor"])


def test_budget_is_respected_on_calibration(trained):
    _, train_ds, _, model = trained
    x, y = train_ds.X[:128], train_ds.y[:128]
    for budget in (0.0, 0.01, 0.1):
        res = auto_mixed_precision(model, x, y, max_acc_drop=budget)
        assert res["calib_float_acc"] - res["calib_acc"] <= budget + 1e-12
        for step in res["path"]:
            assert res["calib_float_acc"] - step["calib_acc"] <= budget + 1e-12
        sizes = [s["nbytes"] for s in res["path"]]
        assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
    with pytest.raises(ValueError):
        auto_mixed_precision(model, x, y, max_acc_drop=-0.1)


def test_looser_budget_never_bigger(trained):
    _, train_ds, _, model = trained
    x, y = train_ds.X[:128], train_ds.y[:128]
    tight = auto_mixed_precision(model, x, y, max_acc_drop=0.0)
    loose = auto_mixed_precision(model, x, y, max_acc_drop=1.0)
    assert loose["nbytes"] <= tight["nbytes"]
    # Budget 1.0 accepts everything → every layer at the bottom rung.
    assert set(loose["assignment"].values()) == {"int2"}


def test_clean_model_beats_uniform_int4_within_budget(demo):
    clean = demo["clean"]
    uni4 = next(p for p in clean["pareto"] if p["config"] == "uniform int4")
    assert clean["nbytes"] <= uni4["nbytes"]
    assert clean["test_acc_drop"] <= 0.05
    assert clean["suggested_skip"] == []  # nothing needs float on the clean toy


def test_dw_outlier_keeps_depthwise_at_higher_precision(demo):
    out = demo["dw_outlier"]
    # Per-tensor int4 on the wide depthwise channel collapses accuracy (round 4);
    # the greedy search must refuse that move and keep dw_w at int8 or float.
    assert out["assignment"]["dw_w"] in ("int8", "float")
    assert any(r["move"].startswith("dw_w") for r in out["rejected"])
    assert out["uniform_skip"]["int4"] == ["dw_w"]
    uni4 = next(p for p in out["pareto"] if p["config"] == "uniform int4")
    assert out["test_acc"] > uni4["acc"] + 0.3


def test_pareto_front():
    pts = [
        {"nbytes": 100, "acc": 0.9},
        {"nbytes": 200, "acc": 0.9},  # dominated (bigger, same acc)
        {"nbytes": 50, "acc": 0.5},
        {"nbytes": 300, "acc": 1.0},
    ]
    assert pareto_front(pts) == [True, False, True, True]


def test_pareto_flags_in_report(demo):
    for rep in demo.values():
        flags = [p["pareto"] for p in rep["pareto"]]
        assert any(flags)
        chosen = rep["pareto"][len(rep["path"]) - 1]
        assert chosen["config"].startswith(f"greedy[{len(rep['path']) - 1}]")


def test_deterministic(trained, demo):
    cfg, train_ds, test_ds, model = trained
    again = run_auto_mixed_precision_demo(cfg, model, train_ds, test_ds)
    assert json.dumps(again, sort_keys=True) == json.dumps(demo, sort_keys=True)


def test_per_channel_granularity_runs(trained):
    cfg, train_ds, test_ds, model = trained
    cfg = copy.deepcopy(cfg)
    cfg["quantize"]["auto_mixed_precision"]["granularity"] = "per_channel"
    out = run_auto_mixed_precision_demo(cfg, model, train_ds, test_ds)
    assert out["clean"]["granularity"] == "per_channel"
    assert out["dw_outlier"]["assignment"]["dw_w"] != "float"  # per-channel absorbs the outlier


def test_format_and_report_integration():
    report = run_before_after()
    amp = report["auto_mixed_precision"]
    assert set(amp) == {"clean", "dw_outlier"}
    text = format_mixed_precision(amp["dw_outlier"], "dw_outlier")
    assert "suggested quantize.skip" in text and "pareto" in text
    json.dumps(report)  # serializable for evals/last_report.json


def test_disabled_by_config():
    cfg = load_config()
    cfg["quantize"]["auto_mixed_precision"]["enabled"] = False
    cfg["distill"]["enabled"] = False
    cfg["qat"]["enabled"] = False
    assert "auto_mixed_precision" not in run_before_after(cfg)


def test_runner_mixed_precision_cli():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals" / "runner.py"), "--mixed-precision",
         "--max-acc-drop", "0.05", "--ladder", "float", "8", "4"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "ladder float/int8/int4" in proc.stdout
    assert "dw_outlier" in proc.stdout
    data = json.loads((ROOT / "evals" / "mixed_precision.json").read_text())
    assert data["clean"]["max_acc_drop"] == 0.05
