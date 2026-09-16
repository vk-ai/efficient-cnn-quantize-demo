"""Post-training affine int8 fake-quant (scale + zero-point)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efficient_cnn.model import TinyEfficientCNN

# Weight tensors to quantize (skip biases — tiny + often asymmetric)
WEIGHT_KEYS = ("stem_w", "dw_w", "pw_w", "fc_w")


@dataclass
class QuantStats:
    bits: int
    scale: dict[str, float]
    zero_point: dict[str, int]
    nbytes_fp64: int
    nbytes_int8: int


def _affine_params(w: np.ndarray, bits: int = 8) -> tuple[float, int]:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    w_min = float(w.min())
    w_max = float(w.max())
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


def quantize_model(model: TinyEfficientCNN, bits: int = 8) -> tuple[TinyEfficientCNN, QuantStats]:
    """Return a copy with fake-quantized weights + size stats (fp64 vs packed int8)."""
    out = model.copy()
    scales: dict[str, float] = {}
    zps: dict[str, int] = {}
    nbytes_fp = 0
    nbytes_i8 = 0
    params = out.named_params()
    for name, arr in params.items():
        nbytes_fp += arr.size * 8  # float64 baseline for the toy
        if name in WEIGHT_KEYS:
            scale, zp = _affine_params(arr, bits=bits)
            scales[name] = scale
            zps[name] = zp
            params[name] = fake_quantize_tensor(arr, scale, zp, bits=bits)
            nbytes_i8 += arr.size * 1  # int8 payload
        else:
            nbytes_i8 += arr.size * 8  # keep biases fp
    out.set_params(params)
    stats = QuantStats(
        bits=bits,
        scale=scales,
        zero_point=zps,
        nbytes_fp64=nbytes_fp,
        nbytes_int8=nbytes_i8,
    )
    return out, stats
