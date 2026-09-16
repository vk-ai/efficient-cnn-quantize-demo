"""Tiny depthwise-separable CNN — edge/efficient vision learning toy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from efficient_cnn.layers import (
    conv2d,
    cross_entropy,
    depthwise_conv2d,
    global_avg_pool,
    linear,
    pointwise_conv2d,
    relu,
    softmax,
)


def _kaiming(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / max(fan_in, 1))
    return rng.normal(0.0, std, size=shape).astype(np.float64)


@dataclass
class TinyEfficientCNN:
    """Stem conv → DW+PW block → GAP → linear classifier.

    Inspired by MobileNet-style depthwise-separable blocks used in edge vision
    (Plumerai-style efficient CNN fundamentals — learning only).
    """

    in_channels: int
    width: int
    n_classes: int
    rng: np.random.Generator = field(repr=False)

    stem_w: np.ndarray = field(init=False)
    stem_b: np.ndarray = field(init=False)
    dw_w: np.ndarray = field(init=False)
    dw_b: np.ndarray = field(init=False)
    pw_w: np.ndarray = field(init=False)
    pw_b: np.ndarray = field(init=False)
    fc_w: np.ndarray = field(init=False)
    fc_b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        w = self.width
        self.stem_w = _kaiming((w, self.in_channels, 3, 3), self.rng)
        self.stem_b = np.zeros(w, dtype=np.float64)
        self.dw_w = _kaiming((w, 1, 3, 3), self.rng)
        self.dw_b = np.zeros(w, dtype=np.float64)
        self.pw_w = _kaiming((w, w, 1, 1), self.rng)
        self.pw_b = np.zeros(w, dtype=np.float64)
        self.fc_w = _kaiming((self.n_classes, w), self.rng)
        self.fc_b = np.zeros(self.n_classes, dtype=np.float64)

    def named_params(self) -> dict[str, np.ndarray]:
        return {
            "stem_w": self.stem_w,
            "stem_b": self.stem_b,
            "dw_w": self.dw_w,
            "dw_b": self.dw_b,
            "pw_w": self.pw_w,
            "pw_b": self.pw_b,
            "fc_w": self.fc_w,
            "fc_b": self.fc_b,
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            setattr(self, k, np.asarray(v, dtype=np.float64))

    def copy(self) -> TinyEfficientCNN:
        other = TinyEfficientCNN(
            self.in_channels, self.width, self.n_classes, np.random.default_rng(0)
        )
        other.set_params({k: v.copy() for k, v in self.named_params().items()})
        return other

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pad so 3x3 keeps spatial size on tiny inputs
        x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        h = relu(conv2d(x, self.stem_w, self.stem_b))
        h = np.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        h = relu(depthwise_conv2d(h, self.dw_w, self.dw_b))
        h = relu(pointwise_conv2d(h, self.pw_w, self.pw_b))
        h = global_avg_pool(h)
        return linear(h, self.fc_w, self.fc_b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x)).argmax(axis=1)

    def loss_acc(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        logits = self.forward(x)
        loss = cross_entropy(logits, y)
        acc = float((logits.argmax(axis=1) == y).mean())
        return loss, acc
