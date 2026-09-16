"""Synthetic tiny images: bright blobs in class-specific quadrants (no downloads)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageDataset:
    """NCHW float64 images in [0, 1] and int64 labels."""

    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _blob_image(
    h: int,
    w: int,
    c: int,
    cls: int,
    n_classes: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place a bright blob in a class-tied quadrant; easy for a tiny CNN."""
    img = rng.normal(0.05, noise, size=(c, h, w)).astype(np.float64)
    q = cls % 4
    y0 = 0 if q in (0, 1) else h // 2
    x0 = 0 if q in (0, 2) else w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    cy = y0 + (h // 4) - 0.5
    cx = x0 + (w // 4) - 0.5
    sigma = max(h, w) / 6.0
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    channel = cls % c
    img[channel] += 1.2 * blob
    # Hard zero outside own quadrant to make classes clearly separable
    mask = np.zeros((h, w), dtype=np.float64)
    y1 = y0 + max(h // 2, 1)
    x1 = x0 + max(w // 2, 1)
    mask[y0:y1, x0:x1] = 1.0
    img[channel] = img[channel] * (0.15 + 0.85 * mask) + 0.9 * blob * mask
    return np.clip(img, 0.0, 1.0)


def make_synthetic(
    n_samples: int,
    height: int,
    width: int,
    channels: int,
    n_classes: int,
    noise: float,
    rng: np.random.Generator,
) -> ImageDataset:
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    y = rng.integers(0, n_classes, size=n_samples)
    X = np.stack(
        [
            _blob_image(height, width, channels, int(lab), n_classes, noise, rng)
            for lab in y
        ],
        axis=0,
    )
    return ImageDataset(X=X.astype(np.float64), y=y.astype(np.int64))


def train_test_split(
    n_train: int,
    n_test: int,
    height: int,
    width: int,
    channels: int,
    n_classes: int,
    noise: float,
    seed: int,
) -> tuple[ImageDataset, ImageDataset]:
    rng = np.random.default_rng(seed)
    train = make_synthetic(n_train, height, width, channels, n_classes, noise, rng)
    test = make_synthetic(n_test, height, width, channels, n_classes, noise, rng)
    return train, test
