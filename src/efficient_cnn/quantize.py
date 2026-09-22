"""Post-training affine int8 fake-quant (scale + zero-point) with calib observers + skip list."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from efficient_cnn.model import TinyEfficientCNN

# Weight tensors to quantize (skip biases — tiny + often asymmetric)
WEIGHT_KEYS = ("stem_w", "dw_w", "pw_w", "fc_w")
VALID_OBSERVERS = ("minmax", "percentile")


@dataclass
class QuantStats:
    bits: int
    scale: dict[str, float]
    zero_point: dict[str, int]
    nbytes_fp64: int
    nbytes_int8: int
    observer: str = "minmax"
    percentile: float | None = None
    skipped: list[str] = field(default_factory=list)
    quantized: list[str] = field(default_factory=list)


def _should_skip(name: str, skip: Sequence[str] | None) -> bool:
    """True if any skip substring matches the param name (e.g. 'fc' → fc_w)."""
    if not skip:
        return False
    return any(str(s) in name for s in skip)


def _clip_for_observer(
    w: np.ndarray,
    observer: str,
    percentile: float,
) -> np.ndarray:
    """
    Teaching PTQ observers on *weights* (fake-quant toy — not TensorRT activation calibrators).

    - minmax: use full tensor range
    - percentile: clip to ±p-th abs quantile (stub histogram/percentile calibrator)
    """
    observer = str(observer).lower()
    if observer not in VALID_OBSERVERS:
        raise ValueError(f"observer must be one of {VALID_OBSERVERS}; got {observer!r}")
    arr = np.asarray(w, dtype=np.float64)
    if observer == "minmax":
        return arr
    p = float(percentile)
    if not 0.0 < p <= 100.0:
        raise ValueError(f"percentile must be in (0, 100]; got {p}")
    # Abs-quantile clip (symmetric stub — common teaching simplification)
    lo = float(np.percentile(arr, 100.0 - p))
    hi = float(np.percentile(arr, p))
    if hi == lo:
        # Fall back to abs percentile of magnitude
        mag = float(np.percentile(np.abs(arr), p))
        lo, hi = -mag, mag
    return np.clip(arr, lo, hi)


def _affine_params(
    w: np.ndarray,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
) -> tuple[float, int]:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    clipped = _clip_for_observer(w, observer, percentile)
    w_min = float(clipped.min())
    w_max = float(clipped.max())
    if w_max == w_min:
        return 1.0, 0
    scale = (w_max - w_min) / float(qmax - qmin)
    # zero_point so 0 maps near q: zp = qmin - w_min/scale
    zp = int(np.round(qmin - w_min / scale))
    zp = int(np.clip(zp, qmin, qmax))
    return scale, zp


def fake_quantize_tensor(w: np.ndarray, scale: float, zero_point: int, bits: int = 8) -> np.ndarray:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    q = np.round(w / scale) + zero_point
    q = np.clip(q, qmin, qmax)
    return (q - zero_point) * scale


def quantize_model(
    model: TinyEfficientCNN,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    skip: Sequence[str] | None = None,
) -> tuple[TinyEfficientCNN, QuantStats]:
    """
    Return a copy with fake-quantized weights + size stats (fp64 vs packed int8).

    Parameters
    ----------
    observer : ``minmax`` | ``percentile``
        Weight-range calibrator stub (not TensorRT/ModelOpt activation observers).
    percentile : used when observer=percentile (e.g. 99.0).
    skip : name substrings to leave in float (e.g. ``["fc"]`` skips the classifier head).
    """
    skip_list = [str(s) for s in (skip or [])]
    out = model.copy()
    scales: dict[str, float] = {}
    zps: dict[str, int] = {}
    nbytes_fp = 0
    nbytes_i8 = 0
    skipped: list[str] = []
    quantized: list[str] = []
    params = out.named_params()
    for name, arr in params.items():
        nbytes_fp += arr.size * 8  # float64 baseline for the toy
        if name in WEIGHT_KEYS:
            if _should_skip(name, skip_list):
                skipped.append(name)
                nbytes_i8 += arr.size * 8  # remain float
                continue
            scale, zp = _affine_params(
                arr, bits=bits, observer=observer, percentile=percentile
            )
            scales[name] = scale
            zps[name] = zp
            params[name] = fake_quantize_tensor(arr, scale, zp, bits=bits)
            nbytes_i8 += arr.size * 1  # int8 payload
            quantized.append(name)
        else:
            nbytes_i8 += arr.size * 8  # keep biases fp
    out.set_params(params)
    stats = QuantStats(
        bits=bits,
        scale=scales,
        zero_point=zps,
        nbytes_fp64=nbytes_fp,
        nbytes_int8=nbytes_i8,
        observer=str(observer).lower(),
        percentile=float(percentile) if str(observer).lower() == "percentile" else None,
        skipped=skipped,
        quantized=quantized,
    )
    return out, stats
