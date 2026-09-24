"""KD + fake-quant QAT tests (CPU, synthetic, fast)."""

from __future__ import annotations

import numpy as np

from efficient_cnn.config import load_config
from efficient_cnn.data import train_test_split
from efficient_cnn.distill import build_teacher_student, run_kd_demo, train_student_with_kd
from efficient_cnn.eval import run_before_after
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.qat import convert_qat, prepare_qat, qat_train, run_qat_demo
from efficient_cnn.train import train


def _tiny_cfg(**overrides):
    cfg = load_config()
    cfg["data"]["n_train"] = 64
    cfg["data"]["n_test"] = 32
    cfg["train"]["epochs"] = 5
    cfg["model"]["width"] = 6
    cfg["distill"] = {
        "enabled": True,
        "temperature": 2.0,
        "alpha": 0.7,
        "student_width": 4,
        "teacher_epochs": 4,
        "epochs": 3,
    }
    cfg["qat"] = {"enabled": True, "steps": 5, "lr": 0.1}
    cfg.update(overrides)
    return cfg


def test_prepare_qat_snaps_weights():
    rng = np.random.default_rng(0)
    model = TinyEfficientCNN(1, 4, 4, rng)
    qmodel, qparams = prepare_qat(model, bits=8)
    assert qparams
    assert "stem_w" in qparams
    # Fake-quant changes values (usually)
    assert qmodel.stem_w.shape == model.stem_w.shape


def test_qat_train_and_convert():
    cfg = _tiny_cfg()
    train_ds, test_ds = train_test_split(
        n_train=64, n_test=32, height=8, width=8, channels=1, n_classes=4, noise=0.05, seed=1
    )
    rng = np.random.default_rng(1)
    model = TinyEfficientCNN(1, 6, 4, rng)
    train(model, train_ds, epochs=3, lr=0.15, batch_size=16, seed=1)
    report = run_qat_demo(model, train_ds, test_ds, cfg)
    assert "ptq" in report and "qat" in report
    assert report["qat"]["steps"] == 5
    assert 0.0 <= report["ptq"]["acc"] <= 1.0
    assert 0.0 <= report["qat"]["acc"] <= 1.0


def test_kd_runs():
    cfg = _tiny_cfg()
    train_ds, test_ds = train_test_split(
        n_train=64, n_test=32, height=8, width=8, channels=1, n_classes=4, noise=0.05, seed=2
    )
    out = run_kd_demo(cfg, train_ds, test_ds)
    assert out["teacher"]["acc"] >= 0.0
    assert out["student_kd"]["acc"] >= 0.0
    assert out["temperature"] == 2.0


def test_eval_includes_kd_and_qat_rows():
    cfg = _tiny_cfg()
    report = run_before_after(cfg)
    assert "distill" in report
    assert "qat_vs_ptq" in report
    assert report["distill"]["student_width"] == 4
