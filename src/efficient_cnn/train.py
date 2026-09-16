"""Tiny SGD trainer with analytical backprop through the efficient CNN."""

from __future__ import annotations

import numpy as np

from efficient_cnn.data import ImageDataset
from efficient_cnn.layers import (
    conv2d,
    depthwise_conv2d,
    global_avg_pool,
    im2col,
    linear,
    pointwise_conv2d,
    relu,
    softmax,
)
from efficient_cnn.model import TinyEfficientCNN


def _relu_mask(pre: np.ndarray) -> np.ndarray:
    return (pre > 0).astype(pre.dtype)


def _train_step(
    model: TinyEfficientCNN,
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
) -> float:
    """One minibatch SGD step; returns mean CE loss."""
    n = x.shape[0]

    # --- forward with caches ---
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    stem_pre = conv2d(x_pad, model.stem_w, model.stem_b)
    stem_act = relu(stem_pre)

    stem_pad = np.pad(stem_act, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    dw_pre = depthwise_conv2d(stem_pad, model.dw_w, model.dw_b)
    dw_act = relu(dw_pre)

    pw_pre = pointwise_conv2d(dw_act, model.pw_w, model.pw_b)
    pw_act = relu(pw_pre)

    gap = global_avg_pool(pw_act)  # (N, C)
    logits = linear(gap, model.fc_w, model.fc_b)
    probs = softmax(logits)

    # loss
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    # --- backward ---
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n

    dfc_w = dlogits.T @ gap
    dfc_b = dlogits.sum(axis=0)
    dgap = dlogits @ model.fc_w  # (N, C)

    # GAP backward: distribute over HxW
    _, c, h, w = pw_act.shape
    dpw_act = dgap.reshape(n, c, 1, 1) / float(h * w)
    dpw_act = np.broadcast_to(dpw_act, pw_act.shape).copy()
    dpw_pre = dpw_act * _relu_mask(pw_pre)

    # pointwise 1x1: weight (out, in, 1, 1)
    # out = einsum over in
    # dpw_w[o,i] = sum_n,h,w dpw_pre[n,o,h,w] * dw_act[n,i,h,w]
    dpw_w = np.einsum("nohw,nihw->oi", dpw_pre, dw_act).reshape(model.pw_w.shape)
    dpw_b = dpw_pre.sum(axis=(0, 2, 3))
    ddw_act = np.einsum("nohw,oi->nihw", dpw_pre, model.pw_w[:, :, 0, 0])

    ddw_pre = ddw_act * _relu_mask(dw_pre)

    # depthwise backward
    ddw_w = np.zeros_like(model.dw_w)
    ddw_b = ddw_pre.sum(axis=(0, 2, 3))
    dstem_pad = np.zeros_like(stem_pad)
    kh, kw = 3, 3
    for ch in range(c):
        cols = im2col(stem_pad[:, ch : ch + 1], kh, kw, 1)  # (N, 9, L)
        dout = ddw_pre[:, ch].reshape(n, 1, -1)  # (N, 1, L)
        # dW: sum over n,L of dout * cols
        ddw_w[ch, 0] = np.einsum("nl,nkl->k", dout[:, 0, :], cols).reshape(kh, kw)
        # dX cols
        w_vec = model.dw_w[ch, 0].ravel()  # (9,)
        dcols = np.einsum("nl,k->nkl", dout[:, 0, :], w_vec)
        # scatter dcols back (valid region matches unpadded stem_act spatial)
        out_h, out_w = stem_act.shape[2], stem_act.shape[3]
        # stem_pad is (N,C,H+2,W+2); im2col sliding on pad
        # Reconstruct gradient into padded tensor then crop
        for i in range(kh):
            for j in range(kw):
                dstem_pad[:, ch, i : i + out_h, j : j + out_w] += dcols[
                    :, i * kw + j, :
                ].reshape(n, out_h, out_w)

    dstem_act = dstem_pad[:, :, 1:-1, 1:-1]
    dstem_pre = dstem_act * _relu_mask(stem_pre)

    # stem conv backward
    in_c = model.in_channels
    cols = im2col(x_pad, 3, 3, 1)  # (N, in*9, L)
    dout = dstem_pre.reshape(n, model.width, -1)
    dstem_w = np.einsum("nol,nkl->ok", dout, cols).reshape(model.stem_w.shape)
    dstem_b = dstem_pre.sum(axis=(0, 2, 3))
    # (no need dX for input)

    # SGD update
    model.fc_w -= lr * dfc_w
    model.fc_b -= lr * dfc_b
    model.pw_w -= lr * dpw_w
    model.pw_b -= lr * dpw_b
    model.dw_w -= lr * ddw_w
    model.dw_b -= lr * ddw_b
    model.stem_w -= lr * dstem_w
    model.stem_b -= lr * dstem_b
    return loss


def train(
    model: TinyEfficientCNN,
    train_ds: ImageDataset,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
) -> list[float]:
    """Shuffle-minibatches SGD; returns per-epoch mean losses."""
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    history: list[float] = []
    for _ in range(epochs):
        idx = rng.permutation(n)
        losses: list[float] = []
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb = train_ds.X[batch]
            yb = train_ds.y[batch]
            losses.append(_train_step(model, xb, yb, lr))
        history.append(float(np.mean(losses)))
    return history
