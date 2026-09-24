"""Fake-quant quantization-aware training (QAT) stub — prepare → train → convert.

Teaching analogue of torchao / PyTorch QAT prepare/convert workflows:
https://docs.pytorch.org/ao/stable/workflows/qat.html
https://pytorch.org/blog/quantization-aware-training/

Uses the same affine fake-quant as PTQ, but inserts it in the forward during
a few SGD steps so the model can adapt. Zero new dependencies.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from efficient_cnn.data import ImageDataset
from efficient_cnn.quantize import (
    WEIGHT_KEYS,
    QuantStats,
    _affine_params,
    fake_quantize_tensor,
    quantize_model,
)
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.train import _train_step


def prepare_qat(
    model: TinyEfficientCNN,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    skip: Sequence[str] | None = None,
) -> tuple[TinyEfficientCNN, dict[str, tuple[float, int]]]:
    """
    Fake-quant weights in-place copy (prepare). Returns model + scale/zp map.

    Biases stay float. Skip list matches PTQ ``quantize.skip``.
    """
    out = model.copy()
    qparams: dict[str, tuple[float, int]] = {}
    skip_list = [str(s) for s in (skip or [])]
    params = out.named_params()
    for name, arr in params.items():
        if name not in WEIGHT_KEYS:
            continue
        if any(s in name for s in skip_list):
            continue
        scale, zp = _affine_params(arr, bits=bits, observer=observer, percentile=percentile)
        qparams[name] = (scale, zp)
        params[name] = fake_quantize_tensor(arr, scale, zp, bits=bits)
    out.set_params(params)
    return out, qparams


def qat_train(
    model: TinyEfficientCNN,
    train_ds: ImageDataset,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    bits: int = 8,
    observer: str = "minmax",
    percentile: float = 99.0,
    skip: Sequence[str] | None = None,
    seed: int = 0,
) -> list[float]:
    """
    Few-step QAT: each step SGD then re-fake-quant weights (STE-style teaching).

    ``steps`` is the number of minibatches (not full epochs) so CI stays fast.
    """
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    losses: list[float] = []
    skip_list = list(skip or [])
    for i in range(int(steps)):
        # Re-prepare fake-quant from current float weights each step
        # (straight-through: update float copy, then snap)
        idx = rng.integers(0, n, size=min(batch_size, n))
        xb = train_ds.X[idx]
        yb = train_ds.y[idx]
        loss = _train_step(model, xb, yb, lr)
        # Snap weights to fake-quant grid
        snapped, _ = prepare_qat(
            model, bits=bits, observer=observer, percentile=percentile, skip=skip_list
        )
        model.set_params(snapped.named_params())
        losses.append(loss)
    return losses


def convert_qat(
    model: TinyEfficientCNN,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    skip: Sequence[str] | None = None,
) -> tuple[TinyEfficientCNN, QuantStats]:
    """Convert after QAT — same packed int8 stats path as PTQ."""
    return quantize_model(
        model, bits=bits, observer=observer, percentile=percentile, skip=skip
    )


def run_qat_demo(
    float_model: TinyEfficientCNN,
    train_ds: ImageDataset,
    test_ds: ImageDataset,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compare PTQ-only vs few-step QAT→convert on the same float checkpoint."""
    q = cfg.get("quantize") or {}
    qat_cfg = cfg.get("qat") or {}
    bits = int(q.get("bits", 8))
    observer = str(q.get("observer", "minmax"))
    percentile = float(q.get("percentile", 99.0))
    skip = list(q.get("skip") or [])
    steps = int(qat_cfg.get("steps", 20))
    lr = float(qat_cfg.get("lr", cfg.get("train", {}).get("lr", 0.05)))
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    seed = int(cfg.get("seed", 0))

    # PTQ baseline from float
    ptq_model, ptq_stats = quantize_model(
        float_model, bits=bits, observer=observer, percentile=percentile, skip=skip
    )
    ptq_loss, ptq_acc = ptq_model.loss_acc(test_ds.X, test_ds.y)

    # QAT path
    qat_model = float_model.copy()
    qat_losses = qat_train(
        qat_model,
        train_ds,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        bits=bits,
        observer=observer,
        percentile=percentile,
        skip=skip,
        seed=seed + 9,
    )
    qat_converted, qat_stats = convert_qat(
        qat_model, bits=bits, observer=observer, percentile=percentile, skip=skip
    )
    qat_loss, qat_acc = qat_converted.loss_acc(test_ds.X, test_ds.y)
    float_loss, float_acc = float_model.loss_acc(test_ds.X, test_ds.y)

    return {
        "float": {"loss": float_loss, "acc": float_acc},
        "ptq": {
            "loss": ptq_loss,
            "acc": ptq_acc,
            "nbytes_int8": ptq_stats.nbytes_int8,
            "nbytes_fp64": ptq_stats.nbytes_fp64,
        },
        "qat": {
            "steps": steps,
            "loss": qat_loss,
            "acc": qat_acc,
            "nbytes_int8": qat_stats.nbytes_int8,
            "final_train_loss": qat_losses[-1] if qat_losses else None,
        },
        "notes": (
            "OSS learning stub — fake-quant QAT vs PTQ; not torchao/TensorRT kernels."
        ),
    }
