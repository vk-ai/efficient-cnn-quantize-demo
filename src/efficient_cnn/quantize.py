"""Post-training affine int8 fake-quant (scale + zero-point) with calib observers + skip list.

Granularity (round 4): ``per_tensor`` (one scale/zero-point per weight tensor, the
original behaviour) or ``per_channel`` (one scale/zero-point per **output
channel**, axis 0). Per-channel is the canonical PTQ fix for depthwise convs,
whose filters can have very different ranges (arXiv 1803.08607, 2004.09602).
Still fake-quant in float: no int8 kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from efficient_cnn.model import TinyEfficientCNN

# Weight tensors to quantize (skip biases — tiny + often asymmetric)
WEIGHT_KEYS = ("stem_w", "dw_w", "pw_w", "fc_w")
VALID_OBSERVERS = ("minmax", "percentile")
VALID_GRANULARITIES = ("per_tensor", "per_channel")
# Bytes to store one (scale, zero_point) pair: fp32 scale + int8 zero-point.
QPARAM_NBYTES = 5


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
    granularity: str = "per_tensor"
    # per_channel: name → per-output-channel scale / zero-point vectors
    channel_scale: dict[str, list[float]] = field(default_factory=dict)
    channel_zero_point: dict[str, list[int]] = field(default_factory=dict)
    # (scale, zp) storage overhead — grows with #channels for per_channel
    nbytes_qparams: int = 0

    @property
    def nbytes_int8_with_qparams(self) -> int:
        return int(self.nbytes_int8 + self.nbytes_qparams)


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


def _check_granularity(granularity: str) -> str:
    g = str(granularity).lower()
    if g not in VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {VALID_GRANULARITIES}; got {granularity!r}"
        )
    return g


def per_channel_params(
    w: np.ndarray,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One affine (scale, zero_point) per slice along ``axis`` (output channel)."""
    arr = np.moveaxis(np.asarray(w, dtype=np.float64), axis, 0)
    scales = np.empty(arr.shape[0], dtype=np.float64)
    zps = np.empty(arr.shape[0], dtype=np.int64)
    for c in range(arr.shape[0]):
        scales[c], zps[c] = _affine_params(
            arr[c], bits=bits, observer=observer, percentile=percentile
        )
    return scales, zps


def fake_quantize_per_channel(
    w: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    bits: int = 8,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Fake-quantize each slice along ``axis`` with its own (scale, zero_point)."""
    arr = np.asarray(w, dtype=np.float64)
    shape = [1] * arr.ndim
    shape[axis] = -1
    s = np.asarray(scales, dtype=np.float64).reshape(shape)
    z = np.asarray(zero_points, dtype=np.float64).reshape(shape)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    q = np.clip(np.round(arr / s) + z, qmin, qmax)
    return (q - z) * s


def quantize_weight(
    w: np.ndarray,
    bits: int = 8,
    *,
    granularity: str = "per_tensor",
    observer: str = "minmax",
    percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fake-quantize one weight tensor. Returns ``(w_q, scales, zero_points)``.

    ``scales`` / ``zero_points`` have length 1 (per_tensor) or ``w.shape[0]``
    (per_channel, one per output channel).
    """
    g = _check_granularity(granularity)
    if g == "per_tensor":
        scale, zp = _affine_params(w, bits=bits, observer=observer, percentile=percentile)
        return (
            fake_quantize_tensor(w, scale, zp, bits=bits),
            np.array([scale], dtype=np.float64),
            np.array([zp], dtype=np.int64),
        )
    scales, zps = per_channel_params(w, bits, observer=observer, percentile=percentile)
    return fake_quantize_per_channel(w, scales, zps, bits=bits), scales, zps


def quantize_model(
    model: TinyEfficientCNN,
    bits: int = 8,
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    skip: Sequence[str] | None = None,
    granularity: str = "per_tensor",
) -> tuple[TinyEfficientCNN, QuantStats]:
    """
    Return a copy with fake-quantized weights + size stats (fp64 vs packed int8).

    Parameters
    ----------
    observer : ``minmax`` | ``percentile``
        Weight-range calibrator stub (not TensorRT/ModelOpt activation observers).
    percentile : used when observer=percentile (e.g. 99.0).
    skip : name substrings to leave in float (e.g. ``["fc"]`` skips the classifier head).
    granularity : ``per_tensor`` (default) | ``per_channel`` (one scale/zp per output
        channel, axis 0). ``stats.scale`` keeps a per-tensor summary (max channel scale)
        and ``stats.channel_scale`` the full vectors.
    """
    g = _check_granularity(granularity)
    skip_list = [str(s) for s in (skip or [])]
    out = model.copy()
    scales: dict[str, float] = {}
    zps: dict[str, int] = {}
    nbytes_fp = 0
    nbytes_i8 = 0
    skipped: list[str] = []
    quantized: list[str] = []
    ch_scale: dict[str, list[float]] = {}
    ch_zp: dict[str, list[int]] = {}
    nbytes_qp = 0
    params = out.named_params()
    for name, arr in params.items():
        nbytes_fp += arr.size * 8  # float64 baseline for the toy
        if name in WEIGHT_KEYS:
            if _should_skip(name, skip_list):
                skipped.append(name)
                nbytes_i8 += arr.size * 8  # remain float
                continue
            wq, s_vec, z_vec = quantize_weight(
                arr, bits=bits, granularity=g, observer=observer, percentile=percentile
            )
            if g == "per_tensor":
                scales[name] = float(s_vec[0])
                zps[name] = int(z_vec[0])
            else:
                scales[name] = float(np.max(s_vec))
                zps[name] = int(z_vec[int(np.argmax(s_vec))])
                ch_scale[name] = [float(v) for v in s_vec]
                ch_zp[name] = [int(v) for v in z_vec]
            params[name] = wq
            nbytes_qp += len(s_vec) * QPARAM_NBYTES
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
        granularity=g,
        channel_scale=ch_scale,
        channel_zero_point=ch_zp,
        nbytes_qparams=nbytes_qp,
    )
    return out, stats


def layer_error_table(
    model: TinyEfficientCNN,
    bits_list: Sequence[int] = (8,),
    *,
    observer: str = "minmax",
    percentile: float = 99.0,
    x: np.ndarray | None = None,
) -> list[dict]:
    """Per-layer weight-reconstruction error, per_tensor vs per_channel.

    One row per (weight tensor, bits):

    - ``mse_per_tensor`` / ``mse_per_channel``: mean((w − Q(w))²)
    - ``rel_mse_*``: MSE / mean(w²), comparable across layers
    - ``ratio``: mse_per_tensor / mse_per_channel (> 1 means per-channel helps)
    - ``channel_range_spread``: max/min per-channel (max−min) range. A big spread
      means one scale per tensor wastes most levels on small channels.
    - ``logit_mse_*`` (when ``x`` is given): MSE of logits vs float when **only
      this layer** is quantized (output-level sensitivity)
    - ``is_depthwise``: True for ``dw_w`` (the layer the table highlights)
    """
    rows: list[dict] = []
    params = model.named_params()
    ref_logits = model.forward(x) if x is not None else None
    for bits in bits_list:
        for name in WEIGHT_KEYS:
            w = params[name]
            ranges = np.ptp(w.reshape(w.shape[0], -1), axis=1)
            spread = float(ranges.max() / max(ranges.min(), 1e-12))
            energy = float(np.mean(w**2)) or 1.0
            row: dict = {
                "layer": name,
                "shape": list(w.shape),
                "bits": int(bits),
                "n_channels": int(w.shape[0]),
                "channel_range_spread": spread,
                "is_depthwise": name == "dw_w",
            }
            for g in VALID_GRANULARITIES:
                wq, _, _ = quantize_weight(
                    w, bits, granularity=g, observer=observer, percentile=percentile
                )
                mse = float(np.mean((w - wq) ** 2))
                row[f"mse_{g}"] = mse
                row[f"rel_mse_{g}"] = mse / energy
                if x is not None:
                    one = model.copy()
                    one.set_params({name: wq})
                    row[f"logit_mse_{g}"] = float(np.mean((one.forward(x) - ref_logits) ** 2))
            row["ratio"] = row["mse_per_tensor"] / max(row["mse_per_channel"], 1e-30)
            rows.append(row)
    return rows


def format_layer_error_table(rows: list[dict]) -> str:
    """Plain-text table; the depthwise row is marked with ``◀ depthwise``."""
    has_logit = bool(rows) and "logit_mse_per_tensor" in rows[0]
    head = (
        f"{'layer':<7} {'bits':>4} {'ch':>3} {'spread':>7} "
        f"{'mse_tensor':>11} {'mse_channel':>11} {'ratio':>6}"
    )
    if has_logit:
        head += f" {'logit_mse_t':>11} {'logit_mse_c':>11}"
    lines = ["Per-layer weight quant error (per_tensor vs per_channel)", head]
    for r in rows:
        line = (
            f"{r['layer']:<7} {r['bits']:>4d} {r['n_channels']:>3d} "
            f"{r['channel_range_spread']:>7.2f} {r['mse_per_tensor']:>11.3e} "
            f"{r['mse_per_channel']:>11.3e} {r['ratio']:>6.1f}"
        )
        if has_logit:
            line += f" {r['logit_mse_per_tensor']:>11.3e} {r['logit_mse_per_channel']:>11.3e}"
        if r["is_depthwise"]:
            line += "  ◀ depthwise"
        lines.append(line)
    return "\n".join(lines)


def rescale_depthwise_channel(
    model: TinyEfficientCNN, channel: int = 0, factor: float = 16.0
) -> TinyEfficientCNN:
    """Function-preserving outlier injection for the depthwise layer (teaching).

    Scale depthwise filter ``channel`` (and its bias) by ``factor`` > 0 and the
    matching pointwise *input* column by ``1/factor``. ReLU is positively
    homogeneous, so the float model's logits are unchanged. The depthwise
    tensor, however, now has one channel with a much wider range, which is the
    situation BN folding creates in real MobileNets (arXiv 1803.08607).
    Per-tensor PTQ suffers and per-channel does not.
    """
    if factor <= 0:
        raise ValueError("factor must be > 0 (ReLU homogeneity needs a positive scale)")
    out = model.copy()
    p = out.named_params()
    dw_w, dw_b, pw_w = p["dw_w"].copy(), p["dw_b"].copy(), p["pw_w"].copy()
    dw_w[channel] *= factor
    dw_b[channel] *= factor
    pw_w[:, channel] /= factor
    out.set_params({"dw_w": dw_w, "dw_b": dw_b, "pw_w": pw_w})
    return out
