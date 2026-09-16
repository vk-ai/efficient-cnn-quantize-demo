"""Before/after eval harness: float baseline vs int8 fake-quant vs magnitude prune."""

from __future__ import annotations

from typing import Any

import numpy as np

from efficient_cnn.config import load_config
from efficient_cnn.data import train_test_split
from efficient_cnn.metrics import estimate_flops, param_count
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.prune import prune_model
from efficient_cnn.quantize import quantize_model
from efficient_cnn.train import train


def run_before_after(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train tiny CNN, then compare float / quantized / pruned on held-out set."""
    cfg = cfg or load_config()
    seed = int(cfg["seed"])
    d = cfg["data"]
    m = cfg["model"]
    t = cfg["train"]
    q = cfg["quantize"]
    p = cfg["prune"]

    train_ds, test_ds = train_test_split(
        n_train=int(d["n_train"]),
        n_test=int(d["n_test"]),
        height=int(d["height"]),
        width=int(d["width"]),
        channels=int(d["channels"]),
        n_classes=int(d["n_classes"]),
        noise=float(d["noise"]),
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    model = TinyEfficientCNN(
        in_channels=int(d["channels"]),
        width=int(m["width"]),
        n_classes=int(d["n_classes"]),
        rng=rng,
    )
    train(
        model,
        train_ds,
        epochs=int(t["epochs"]),
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        seed=seed,
    )

    base_loss, base_acc = model.loss_acc(test_ds.X, test_ds.y)
    base_params = param_count(model)
    base_nnz = param_count(model, nonzero_only=True)
    flops = estimate_flops(model, int(d["height"]), int(d["width"]))

    q_model, q_stats = quantize_model(model, bits=int(q["bits"]))
    q_loss, q_acc = q_model.loss_acc(test_ds.X, test_ds.y)

    p_model, p_stats = prune_model(model, sparsity=float(p["sparsity"]))
    p_loss, p_acc = p_model.loss_acc(test_ds.X, test_ds.y)
    p_nnz = param_count(p_model, nonzero_only=True)

    report = {
        "baseline": {
            "loss": base_loss,
            "acc": base_acc,
            "params": base_params,
            "nonzero": base_nnz,
            "flops": flops,
            "nbytes_fp64": q_stats.nbytes_fp64,
        },
        "quantized": {
            "bits": q_stats.bits,
            "loss": q_loss,
            "acc": q_acc,
            "nbytes_int8_pack": q_stats.nbytes_int8,
            "size_ratio": q_stats.nbytes_int8 / max(q_stats.nbytes_fp64, 1),
            "acc_drop": base_acc - q_acc,
        },
        "pruned": {
            "sparsity": p_stats.sparsity,
            "loss": p_loss,
            "acc": p_acc,
            "nonzero": p_nnz,
            "pruned_weights": p_stats.pruned,
            "acc_drop": base_acc - p_acc,
        },
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    b, q, p = report["baseline"], report["quantized"], report["pruned"]
    lines = [
        "Efficient CNN quantize/prune eval",
        f"  baseline  loss={b['loss']:.4f}  acc={b['acc']:.3f}  "
        f"params={b['params']}  flops≈{b['flops']}  nbytes_fp64={b['nbytes_fp64']}",
        f"  int8 fq   loss={q['loss']:.4f}  acc={q['acc']:.3f}  "
        f"nbytes≈{q['nbytes_int8_pack']}  size_ratio={q['size_ratio']:.3f}  "
        f"acc_drop={q['acc_drop']:+.3f}",
        f"  prune     loss={p['loss']:.4f}  acc={p['acc']:.3f}  "
        f"nonzero={p['nonzero']}  pruned={p['pruned_weights']}  "
        f"acc_drop={p['acc_drop']:+.3f}",
    ]
    return "\n".join(lines)
