"""Sensitivity-driven automatic skip list / mixed precision (round 5, numpy teaching toy).

Round 4's ``layer_error_table`` *measured* per-layer sensitivity: the logit MSE
when only one layer is quantized. This module turns that measurement into a
*decision*, in the spirit of NVIDIA ModelOpt AutoQuantize, AIMET AMP and the HF
sensitivity-aware mixed-precision blog:

1. **Measure.** For every weight tensor and every precision in the ladder
   (default ``float → int8 → int4 → int2``), quantize *only that layer* and record
   calibration accuracy and logit MSE vs float (the R4 definition).
2. **Greedy descent.** Start all-float. Repeatedly take the cheapest step: lower
   one layer by one rung, choosing the step with the smallest measured
   sensitivity per byte saved. Evaluate the *whole* mixed model on the
   calibration set and keep the step only if the accuracy drop vs float stays within
   ``max_acc_drop``. Otherwise that layer stays at its current precision.
3. **Report** the per-layer assignment, the suggested ``quantize.skip`` (layers
   left in float), the skip lists for a uniform int8 / int4 run, and a Pareto table
   (bytes vs held-out accuracy) of the greedy path plus the uniform baselines.

Decisions use a **calibration split** (a slice of the training data). The test
set only scores the result, so the budget is not tuned on the numbers you report.

Weight-only fake-quant in float, like the rest of the repo. Sizes use the toy's
conventions: float = fp64 (8 B per element), intN = N/8 B per element (packed),
plus 5 B per (scale, zero-point), with biases kept in fp64. These are not int4/int2
kernels, and this is not ModelOpt, AIMET or TensorRT.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.quantize import QPARAM_NBYTES, WEIGHT_KEYS, quantize_weight

FLOAT = "float"
DEFAULT_LADDER: tuple[Any, ...] = (FLOAT, 8, 4, 2)
Precision = Any  # "float" | int bits


def _norm_precision(p: Any) -> Precision:
    if isinstance(p, str) and p.lower() in ("float", "fp", "fp64", "none"):
        return FLOAT
    if isinstance(p, str) and p.lower().startswith("int"):
        p = p[3:]
    bits = int(p)
    if not 2 <= bits <= 16:
        raise ValueError(f"bits must be in [2, 16] or 'float'; got {p!r}")
    return bits


def normalize_ladder(ladder: Sequence[Any]) -> list[Precision]:
    """Validate a precision ladder: starts at ``float``, then strictly decreasing bits."""
    out = [_norm_precision(p) for p in ladder]
    if not out or out[0] != FLOAT:
        raise ValueError("ladder must start with 'float'")
    bits = out[1:]
    if not bits:
        raise ValueError("ladder needs at least one integer precision after 'float'")
    if any(b == FLOAT for b in bits) or any(a <= b for a, b in zip(bits, bits[1:])):
        raise ValueError(f"ladder bits must be strictly decreasing; got {out}")
    return out


def _label(p: Precision) -> str:
    return "float" if p == FLOAT else f"int{p}"


def layer_nbytes(model: TinyEfficientCNN, name: str, p: Precision, granularity: str) -> float:
    """Storage for one weight tensor at precision ``p`` (toy conventions, see module doc)."""
    w = model.named_params()[name]
    if p == FLOAT:
        return float(w.size * 8)
    n_qparams = 1 if granularity == "per_tensor" else int(w.shape[0])
    return float(w.size * int(p) / 8.0 + n_qparams * QPARAM_NBYTES)


def model_nbytes(
    model: TinyEfficientCNN, assignment: dict[str, Precision], granularity: str
) -> float:
    total = 0.0
    for name, arr in model.named_params().items():
        if name in WEIGHT_KEYS:
            total += layer_nbytes(model, name, assignment.get(name, FLOAT), granularity)
        else:
            total += arr.size * 8  # biases stay fp64
    return total


def quantize_mixed(
    model: TinyEfficientCNN,
    assignment: dict[str, Precision],
    *,
    granularity: str = "per_tensor",
    observer: str = "minmax",
    percentile: float = 99.0,
) -> TinyEfficientCNN:
    """Copy of ``model`` with each weight tensor fake-quantized at its own precision."""
    unknown = set(assignment) - set(WEIGHT_KEYS)
    if unknown:
        raise KeyError(f"unknown layers {sorted(unknown)}; expected {WEIGHT_KEYS}")
    out = model.copy()
    params = out.named_params()
    new = {}
    for name in WEIGHT_KEYS:
        p = _norm_precision(assignment.get(name, FLOAT))
        if p != FLOAT:
            new[name], _, _ = quantize_weight(
                params[name], p, granularity=granularity, observer=observer,
                percentile=percentile,
            )
    out.set_params(new)
    return out


def measure_sensitivity(
    model: TinyEfficientCNN,
    x: np.ndarray,
    y: np.ndarray,
    ladder: Sequence[Any] = DEFAULT_LADDER,
    *,
    granularity: str = "per_tensor",
    observer: str = "minmax",
    percentile: float = 99.0,
) -> list[dict[str, Any]]:
    """One row per (layer, integer precision): quantize only that layer, measure on (x, y)."""
    ladder = normalize_ladder(ladder)
    ref = model.forward(x)
    _, float_acc = model.loss_acc(x, y)
    rows = []
    for name in WEIGHT_KEYS:
        for p in ladder[1:]:
            one = quantize_mixed(
                model, {name: p}, granularity=granularity, observer=observer,
                percentile=percentile,
            )
            _, acc = one.loss_acc(x, y)
            rows.append(
                {
                    "layer": name,
                    "precision": _label(p),
                    "bits": int(p),
                    "logit_mse": float(np.mean((one.forward(x) - ref) ** 2)),
                    "acc": float(acc),
                    "acc_drop": float(float_acc - acc),
                    "nbytes": layer_nbytes(model, name, p, granularity),
                }
            )
    return rows


def auto_mixed_precision(
    model: TinyEfficientCNN,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    max_acc_drop: float = 0.01,
    ladder: Sequence[Any] = DEFAULT_LADDER,
    granularity: str = "per_tensor",
    observer: str = "minmax",
    percentile: float = 99.0,
) -> dict[str, Any]:
    """Greedy sensitivity-per-byte descent under an accuracy budget (see module doc)."""
    if max_acc_drop < 0:
        raise ValueError("max_acc_drop must be >= 0")
    ladder = normalize_ladder(ladder)
    qkw = dict(granularity=granularity, observer=observer, percentile=percentile)
    sens = measure_sensitivity(model, x_calib, y_calib, ladder, **qkw)
    sens_by = {(r["layer"], r["precision"]): r for r in sens}
    _, float_acc = model.loss_acc(x_calib, y_calib)

    assignment: dict[str, Precision] = {name: FLOAT for name in WEIGHT_KEYS}
    frozen: set[str] = set()
    path: list[dict[str, Any]] = [
        {
            "step": 0,
            "move": "start (all float)",
            "assignment": {k: _label(v) for k, v in assignment.items()},
            "nbytes": model_nbytes(model, assignment, granularity),
            "calib_acc": float(float_acc),
        }
    ]
    rejected: list[dict[str, Any]] = []
    while True:
        candidates = []
        for name in WEIGHT_KEYS:
            if name in frozen:
                continue
            idx = ladder.index(assignment[name])
            if idx + 1 >= len(ladder):
                continue
            nxt = ladder[idx + 1]
            saved = layer_nbytes(model, name, assignment[name], granularity) - layer_nbytes(
                model, name, nxt, granularity
            )
            if saved <= 0:
                frozen.add(name)  # e.g. per_channel qparams outweigh the savings
                continue
            cost = sens_by[(name, _label(nxt))]["logit_mse"]
            # Stable, deterministic order: cost per byte, then layer order.
            candidates.append((cost / saved, WEIGHT_KEYS.index(name), name, nxt, saved))
        if not candidates:
            break
        _, _, name, nxt, saved = min(candidates)
        trial = {**assignment, name: nxt}
        _, acc = quantize_mixed(model, trial, **qkw).loss_acc(x_calib, y_calib)
        move = f"{name}: {_label(assignment[name])} → {_label(nxt)}"
        if float_acc - acc <= max_acc_drop + 1e-12:
            assignment = trial
            path.append(
                {
                    "step": len(path),
                    "move": move,
                    "assignment": {k: _label(v) for k, v in assignment.items()},
                    "nbytes": model_nbytes(model, assignment, granularity),
                    "calib_acc": float(acc),
                }
            )
        else:
            frozen.add(name)
            rejected.append({"move": move, "calib_acc": float(acc),
                             "acc_drop": float(float_acc - acc)})
    final = path[-1]
    return {
        "granularity": granularity,
        "observer": observer,
        "ladder": [_label(p) for p in ladder],
        "max_acc_drop": float(max_acc_drop),
        "calib_float_acc": float(float_acc),
        "sensitivity": sens,
        "assignment": dict(final["assignment"]),
        "bits_per_layer": {k: (None if v == FLOAT else int(v)) for k, v in assignment.items()},
        "suggested_skip": [k for k, v in assignment.items() if v == FLOAT],
        "nbytes": final["nbytes"],
        "calib_acc": final["calib_acc"],
        "path": path,
        "rejected": rejected,
    }


def pareto_front(points: Sequence[dict[str, Any]]) -> list[bool]:
    """Non-dominated flags for (``nbytes`` lower-is-better, ``acc`` higher-is-better)."""
    flags = []
    for i, p in enumerate(points):
        dominated = any(
            q["nbytes"] <= p["nbytes"]
            and q["acc"] >= p["acc"]
            and (q["nbytes"] < p["nbytes"] or q["acc"] > p["acc"])
            for j, q in enumerate(points)
            if j != i
        )
        flags.append(not dominated)
    return flags


def run_auto_mixed_precision(
    model: TinyEfficientCNN,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_acc_drop: float = 0.01,
    ladder: Sequence[Any] = DEFAULT_LADDER,
    granularity: str = "per_tensor",
    observer: str = "minmax",
    percentile: float = 99.0,
) -> dict[str, Any]:
    """Greedy selection on calibration data + held-out scoring + Pareto table."""
    ladder_n = normalize_ladder(ladder)
    qkw = dict(granularity=granularity, observer=observer, percentile=percentile)
    res = auto_mixed_precision(
        model, x_calib, y_calib, max_acc_drop=max_acc_drop, ladder=ladder_n, **qkw
    )
    float_bytes = model_nbytes(model, {}, granularity)

    def _score(label: str, assignment: dict[str, Precision], kind: str) -> dict[str, Any]:
        loss, acc = quantize_mixed(model, assignment, **qkw).loss_acc(x_test, y_test)
        nbytes = model_nbytes(model, assignment, granularity)
        return {
            "config": label,
            "kind": kind,
            "assignment": {k: _label(_norm_precision(assignment.get(k, FLOAT)))
                           for k in WEIGHT_KEYS},
            "nbytes": nbytes,
            "size_ratio": nbytes / float_bytes,
            "acc": float(acc),
            "loss": float(loss),
        }

    points = []
    for step in res["path"]:
        a = {k: _norm_precision(v) for k, v in step["assignment"].items()}
        points.append(_score(f"greedy[{step['step']}] {step['move']}", a, "greedy"))
    for p in ladder_n[1:]:
        points.append(_score(f"uniform {_label(p)}", {k: p for k in WEIGHT_KEYS}, "uniform"))
    # Suggested skip list if you keep the repo's *uniform* quantize.bits pipeline.
    uniform_skip: dict[str, list[str]] = {}
    for p in ladder_n[1:]:
        if p not in (8, 4):
            continue
        one = auto_mixed_precision(
            model, x_calib, y_calib, max_acc_drop=max_acc_drop, ladder=[FLOAT, p], **qkw
        )
        uniform_skip[_label(p)] = one["suggested_skip"]
        if one["suggested_skip"]:
            a = {k: (FLOAT if k in one["suggested_skip"] else p) for k in WEIGHT_KEYS}
            points.append(
                _score(f"{_label(p)} skip={one['suggested_skip']}", a, "uniform+skip")
            )
    for pt, flag in zip(points, pareto_front(points)):
        pt["pareto"] = flag
    _, test_float_acc = model.loss_acc(x_test, y_test)
    chosen = points[len(res["path"]) - 1]
    return {
        **res,
        "test_float_acc": float(test_float_acc),
        "test_acc": chosen["acc"],
        "test_acc_drop": float(test_float_acc - chosen["acc"]),
        "float_nbytes": float_bytes,
        "size_ratio": res["nbytes"] / float_bytes,
        "uniform_skip": uniform_skip,
        "pareto": points,
    }


def format_mixed_precision(report: dict[str, Any], title: str = "") -> str:
    a = report["assignment"]
    lines = [
        f"Auto mixed precision{(' — ' + title) if title else ''} "
        f"(greedy on calib, budget acc_drop ≤ {report['max_acc_drop']:.3f}, "
        f"{report['granularity']}, ladder {'/'.join(report['ladder'])})",
        "  assignment: " + "  ".join(f"{k}={v}" for k, v in a.items()),
        f"  calib acc {report['calib_acc']:.3f} (float {report['calib_float_acc']:.3f})  "
        f"test acc {report['test_acc']:.3f} (float {report['test_float_acc']:.3f})  "
        f"bytes {report['nbytes']:.0f}/{report['float_nbytes']:.0f} "
        f"(size_ratio {report['size_ratio']:.3f})",
        f"  suggested quantize.skip (mixed): {report['suggested_skip']}   "
        + "  ".join(f"uniform {k} → skip {v}" for k, v in report["uniform_skip"].items()),
    ]
    if report["rejected"]:
        lines.append(
            "  rejected: " + "; ".join(
                f"{r['move']} (calib acc {r['calib_acc']:.3f})" for r in report["rejected"]
            )
        )
    lines.append(f"  {'config':<44} {'bytes':>7} {'ratio':>6} {'test_acc':>8}  pareto")
    for p in report["pareto"]:
        lines.append(
            f"  {p['config'][:44]:<44} {p['nbytes']:>7.0f} {p['size_ratio']:>6.3f} "
            f"{p['acc']:>8.3f}  {'*' if p['pareto'] else ''}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_LADDER",
    "normalize_ladder",
    "layer_nbytes",
    "model_nbytes",
    "quantize_mixed",
    "measure_sensitivity",
    "auto_mixed_precision",
    "pareto_front",
    "run_auto_mixed_precision",
    "format_mixed_precision",
]
