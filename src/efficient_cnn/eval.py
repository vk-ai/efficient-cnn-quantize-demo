"""Before/after eval harness: float baseline vs int8 fake-quant vs magnitude prune vs compose."""

from __future__ import annotations

from typing import Any

import numpy as np

from efficient_cnn.config import load_config
from efficient_cnn.data import train_test_split
from efficient_cnn.metrics import estimate_flops, param_count
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.prune import prune_model
from efficient_cnn.quantize import (
    VALID_GRANULARITIES,
    format_layer_error_table,
    layer_error_table,
    quantize_model,
    rescale_depthwise_channel,
)
from efficient_cnn.train import train
from efficient_cnn.distill import run_kd_demo
from efficient_cnn.qat import run_qat_demo

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
    granularity: str = "per_tensor",
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
            current, q_stats = quantize_model(current, bits=bits, granularity=granularity)
            meta["steps"]["quantize"] = {
                "bits": q_stats.bits,
                "nbytes_fp64": q_stats.nbytes_fp64,
                "nbytes_int8": q_stats.nbytes_int8,
                "observer": q_stats.observer,
                "granularity": q_stats.granularity,
                "skipped": list(q_stats.skipped),
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

    bits = int(q["bits"])
    observer = str(q.get("observer", "minmax"))
    percentile = float(q.get("percentile", 99.0))
    skip = list(q.get("skip") or [])
    granularity = str(q.get("granularity", "per_tensor")).lower()
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"quantize.granularity must be one of {VALID_GRANULARITIES}; got {granularity!r}"
        )

    def _quant_row(
        obs: str,
        skip_list: list[str],
        *,
        label_bits: bool = True,
        gran: str = "per_tensor",
        n_bits: int | None = None,
        src: TinyEfficientCNN | None = None,
    ) -> tuple:
        use_bits = bits if n_bits is None else int(n_bits)
        qm, qs = quantize_model(
            model if src is None else src,
            bits=use_bits,
            observer=obs,
            percentile=percentile,
            skip=skip_list,
            granularity=gran,
        )
        loss, acc = qm.loss_acc(test_ds.X, test_ds.y)
        row = {
            "bits": qs.bits if label_bits else use_bits,
            "observer": qs.observer,
            "granularity": qs.granularity,
            "percentile": qs.percentile,
            "skip": list(skip_list),
            "skipped": list(qs.skipped),
            "quantized": list(qs.quantized),
            "loss": loss,
            "acc": acc,
            "nbytes_int8_pack": qs.nbytes_int8,
            "nbytes_qparams": qs.nbytes_qparams,
            "size_ratio": qs.nbytes_int8 / max(qs.nbytes_fp64, 1),
            "acc_drop": base_acc - acc,
        }
        return qm, qs, row

    # Default quantized row = configured observer + skip (backward-compatible key)
    _, q_stats, quantized_row = _quant_row(observer, skip, gran=granularity)
    _, _, int8_minmax_row = _quant_row("minmax", [])
    _, _, int8_percentile_row = _quant_row("percentile", [])
    # Skip-head teaching row: leave classifier (fc*) in float
    skip_head = skip if skip else ["fc"]
    _, _, int8_skip_head_row = _quant_row("minmax", skip_head)

    # Round 4: granularity ablation (minmax, no skip) + optional int4 rows
    gran_rows: dict[str, dict[str, Any]] = {}
    bits_list = [8] + ([4] if q.get("int4_rows", True) else [])
    for nb in bits_list:
        for g in VALID_GRANULARITIES:
            _, _, gran_rows[f"int{nb}_{g}"] = _quant_row("minmax", [], gran=g, n_bits=nb)
    x_probe = test_ds.X[: min(64, len(test_ds))]
    err_rows = (
        layer_error_table(model, bits_list, x=x_probe) if q.get("error_table", True) else []
    )
    outlier: dict[str, Any] | None = None
    factor = float(q.get("dw_outlier_factor", 0) or 0)
    if factor > 0:
        # Function-preserving: same float logits, one wide depthwise channel.
        om = rescale_depthwise_channel(model, channel=0, factor=factor)
        o_loss, o_acc = om.loss_acc(test_ds.X, test_ds.y)
        outlier = {"factor": factor, "channel": 0, "float_acc": o_acc, "float_loss": o_loss}
        for nb in bits_list:
            for g in VALID_GRANULARITIES:
                _, _, r = _quant_row("minmax", [], gran=g, n_bits=nb, src=om)
                outlier[f"int{nb}_{g}"] = {"acc": r["acc"], "loss": r["loss"]}
        outlier["layer_error_table"] = [
            r for r in layer_error_table(om, bits_list, x=x_probe) if r["is_depthwise"]
        ]

    p_model, p_stats = prune_model(model, sparsity=float(p["sparsity"]))
    p_loss, p_acc = p_model.loss_acc(test_ds.X, test_ds.y)
    p_nnz = param_count(p_model, nonzero_only=True)

    c_model, c_meta = apply_compose(
        model,
        order,
        bits=bits,
        sparsity=float(p["sparsity"]),
        granularity=granularity,
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
        "quantized": quantized_row,
        "int8_minmax": int8_minmax_row,
        "int8_percentile": int8_percentile_row,
        "int8_skip_head": int8_skip_head_row,
        "granularity_ablation": gran_rows,
        "layer_error_table": err_rows,
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
        "quantize_config": {
            "bits": bits,
            "observer": observer,
            "percentile": percentile,
            "skip": skip,
            "granularity": granularity,
        },
    }
    if outlier is not None:
        report["dw_outlier"] = outlier
    distill_cfg = cfg.get("distill") or {}
    if distill_cfg.get("enabled", False):
        report["distill"] = run_kd_demo(cfg, train_ds, test_ds)
    qat_cfg = cfg.get("qat") or {}
    if qat_cfg.get("enabled", False):
        report["qat_vs_ptq"] = run_qat_demo(model, train_ds, test_ds, cfg)
    return report


def format_report(report: dict[str, Any]) -> str:
    b = report["baseline"]
    q = report["quantized"]
    p = report["pruned"]
    c = report["prune_then_int8"]
    mm = report["int8_minmax"]
    pc = report["int8_percentile"]
    sk = report["int8_skip_head"]
    lines = [
        "Efficient CNN quantize/prune eval",
        f"  baseline        loss={b['loss']:.4f}  acc={b['acc']:.3f}  "
        f"params={b['params']}  flops≈{b['flops']}  nbytes_fp64={b['nbytes_fp64']}",
        f"  int8_minmax     loss={mm['loss']:.4f}  acc={mm['acc']:.3f}  "
        f"nbytes≈{mm['nbytes_int8_pack']}  size_ratio={mm['size_ratio']:.3f}  "
        f"acc_drop={mm['acc_drop']:+.3f}",
        f"  int8_percentile loss={pc['loss']:.4f}  acc={pc['acc']:.3f}  "
        f"nbytes≈{pc['nbytes_int8_pack']}  size_ratio={pc['size_ratio']:.3f}  "
        f"acc_drop={pc['acc_drop']:+.3f}  p={pc['percentile']}",
        f"  int8_skip_head  loss={sk['loss']:.4f}  acc={sk['acc']:.3f}  "
        f"nbytes≈{sk['nbytes_int8_pack']}  skip={sk['skip']}  skipped={sk['skipped']}  "
        f"acc_drop={sk['acc_drop']:+.3f}",
        f"  int8 fq (cfg)   loss={q['loss']:.4f}  acc={q['acc']:.3f}  "
        f"observer={q['observer']}  nbytes≈{q['nbytes_int8_pack']}  "
        f"size_ratio={q['size_ratio']:.3f}  acc_drop={q['acc_drop']:+.3f}",
        f"  prune           loss={p['loss']:.4f}  acc={p['acc']:.3f}  "
        f"nonzero={p['nonzero']}  pruned={p['pruned_weights']}  "
        f"acc_drop={p['acc_drop']:+.3f}",
        f"  compose         order={'→'.join(c['order'])}  loss={c['loss']:.4f}  "
        f"acc={c['acc']:.3f}  nonzero={c['nonzero']}  "
        f"nbytes≈{c['nbytes_int8_pack']}  size_ratio={c['size_ratio']:.3f}  "
        f"acc_drop={c['acc_drop']:+.3f}",
    ]
    for key, r in (report.get("granularity_ablation") or {}).items():
        lines.append(
            f"  {key:<16} loss={r['loss']:.4f}  acc={r['acc']:.3f}  "
            f"qparams={r['nbytes_qparams']}B  acc_drop={r['acc_drop']:+.3f}"
        )
    if report.get("layer_error_table"):
        lines.append(format_layer_error_table(report["layer_error_table"]))
    if "dw_outlier" in report:
        o = report["dw_outlier"]
        accs = "  ".join(
            f"{k}={v['acc']:.3f}" for k, v in o.items() if k.startswith("int")
        )
        lines.append(
            f"  dw_outlier x{o['factor']:g} (float_acc={o['float_acc']:.3f}, same logits)  {accs}"
        )
    if "distill" in report:
        d = report["distill"]
        lines.append(
            f"  distill        teacher_acc={d['teacher']['acc']:.3f}  "
            f"student_solo={d['student_solo']['acc']:.3f}  "
            f"student_kd={d['student_kd']['acc']:.3f}  "
            f"T={d['temperature']}  alpha={d['alpha']}"
        )
    if "qat_vs_ptq" in report:
        qv = report["qat_vs_ptq"]
        lines.append(
            f"  qat_vs_ptq     float_acc={qv['float']['acc']:.3f}  "
            f"ptq_acc={qv['ptq']['acc']:.3f}  "
            f"qat_acc={qv['qat']['acc']:.3f}  "
            f"qat_steps={qv['qat']['steps']}"
        )
    return "\n".join(lines)
