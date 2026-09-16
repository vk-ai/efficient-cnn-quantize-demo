"""Minimal numpy CNN ops: conv2d, depthwise, pointwise, ReLU, GAP, linear."""

from __future__ import annotations

import numpy as np


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1) -> np.ndarray:
    """NCHW -> (N, C*kh*kw, out_h*out_w)."""
    n, c, h, w = x.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[
                :, :, i : i + stride * out_h : stride, j : j + stride * out_w : stride
            ]
    return cols.reshape(n, c * kh * kw, out_h * out_w)


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Standard conv: weight (out_c, in_c, kh, kw), x NCHW."""
    out_c, in_c, kh, kw = weight.shape
    n, c, h, w = x.shape
    assert c == in_c
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    cols = im2col(x, kh, kw, stride)  # (N, in*kh*kw, oh*ow)
    w_mat = weight.reshape(out_c, -1)
    out = w_mat @ cols  # (out_c, N, oh*ow) via broadcast? No — need loop or einsum
    # w_mat (out_c, K) @ cols (N, K, L) -> (N, out_c, L)
    out = np.einsum("ok,nkl->nol", w_mat, cols)
    out = out.reshape(n, out_c, out_h, out_w)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def depthwise_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Depthwise: weight (c, 1, kh, kw)."""
    c, one, kh, kw = weight.shape
    assert one == 1
    n, xc, h, w = x.shape
    assert xc == c
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for ch in range(c):
        cols = im2col(x[:, ch : ch + 1], kh, kw, stride)  # (N, kh*kw, L)
        w_vec = weight[ch, 0].reshape(1, -1)  # (1, kh*kw)
        o = w_vec @ cols  # (N, 1, L) — w_vec @ cols: (1,K) @ (N,K,L)
        o = np.einsum("k,nkl->nl", weight[ch, 0].ravel(), cols)
        out[:, ch] = o.reshape(n, out_h, out_w)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def pointwise_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """1x1 conv: weight (out_c, in_c, 1, 1)."""
    return conv2d(x, weight, bias, stride=1)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """NCHW -> (N, C)."""
    return x.mean(axis=(2, 3))


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """x (N, in), weight (out, in)."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean CE; numerically stable."""
    n = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(z).sum(axis=1) + 1e-12)
    return float((-z[np.arange(n), y] + log_sum).mean())
