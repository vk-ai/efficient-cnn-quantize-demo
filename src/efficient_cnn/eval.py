"""Before/after eval harness: float baseline vs int8 fake-quant vs magnitude prune vs compose."""

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

DEFAULT_COMPOSE_ORDER = ("prune", "quantize")


def _compose_order(cfg: dict[str, Any]) -> list[str]:
    compose = cfg.get("compose") or {}
    order = compose.get("order") if isinstance(compose, dict) else None
    if not order:
        return list(DEFAULT_COMPOSE_ORDER)
    if not isinstance(order, (list, tuple)) or not order:
        raise ValueError("compose.order must be a non-empty list, e.g. [prune, quantize]")
    allowed = {"prune", "quantize"}
    out = [str(step).lower() for step in order]
    unknown = [s for s in out if s not in allowed]
    if unknown:
        raise ValueError(f"compose.order unknown steps {unknown}; allowed={sorted(allowed)}")
    return out


def apply_compose(
    model: TinyEfficientCNN,
    order: list[str],
    *,
    bits: int,
    sparsity: float,
) -> tuple[TinyEfficientCNN, dict[str, Any]]:
    """
    Apply compression steps in order (teaching toy — fake-quant, not real INT8 kernels).

    Default community-oriented recipe: magnitude prune → int8 fake-quant on remaining weights.
    """
    current = model
    meta: dict[str, Any] = {"order": list(order), "steps": {}}
    for step in order:
        if step == "prune":
            current, p_stats = prune_model(current, sparsity=sparsity)
            meta["steps"]["prune"] = {
                "sparsity": p_stats.sparsity,
                "pruned_weights": p_stats.pruned,
                "nonzero_after": p_stats.nonzero_after,
            }
        elif step == "quantize":
            current, q_stats = quantize_model(current, bits=bits)
            meta["steps"]["quantize"] = {
                "bits": q_stats.bits,
                "nbytes_fp64": q_stats.nbytes_fp64,
                "nbytes_int8": q_stats.nbytes_int8,
            }
        else:  # pragma: no cover - validated in _compose_order
            raise ValueError(f"unknown compose step: {step}")
    return current, meta


def run_before_after(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train tiny CNN, then compare float / quantized / pruned / prune→int8 on held-out set."""
    cfg = cfg or load_config()
    seed = int(cfg["seed"])
    d = cfg["data"]
    m = cfg["model"]
    t = cfg["train"]
    q = cfg["quantize"]
    p = cfg["prune"]
    order = _compose_order(cfg)

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

    c_model, c_meta = apply_compose(
        model,
        order,
        bits=int(q["bits"]),
        sparsity=float(p["sparsity"]),
    )
    c_loss, c_acc = c_model.loss_acc(test_ds.X, test_ds.y)
    c_nnz = param_count(c_model, nonzero_only=True)
    q_step = c_meta["steps"].get("quantize", {})
    nbytes_int8 = int(q_step.get("nbytes_int8", q_stats.nbytes_int8))
    nbytes_fp64 = int(q_step.get("nbytes_fp64", q_stats.nbytes_fp64))

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
        "prune_then_int8": {
            "order": c_meta["order"],
            "loss": c_loss,
            "acc": c_acc,
            "nonzero": c_nnz,
            "nbytes_int8_pack": nbytes_int8,
            "size_ratio": nbytes_int8 / max(nbytes_fp64, 1),
            "acc_drop": base_acc - c_acc,
            "steps": c_meta["steps"],
        },
        "compose": {"order": c_meta["order"]},
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    b, q, p, c = (
        report["baseline"],
        report["quantized"],
        report["pruned"],
        report["prune_then_int8"],
    )
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
        f"  compose   order={'→'.join(c['order'])}  loss={c['loss']:.4f}  "
        f"acc={c['acc']:.3f}  nonzero={c['nonzero']}  "
        f"nbytes≈{c['nbytes_int8_pack']}  size_ratio={c['size_ratio']:.3f}  "
        f"acc_drop={c['acc_drop']:+.3f}",
    ]
    return "\n".join(lines)
