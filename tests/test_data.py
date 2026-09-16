import numpy as np

from efficient_cnn.data import make_synthetic, train_test_split


def test_synthetic_shapes():
    rng = np.random.default_rng(0)
    ds = make_synthetic(32, 8, 8, 1, 4, 0.1, rng)
    assert ds.X.shape == (32, 1, 8, 8)
    assert ds.y.shape == (32,)
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0


def test_train_test_split():
    train, test = train_test_split(64, 32, 8, 8, 1, 4, 0.15, seed=1)
    assert len(train) == 64 and len(test) == 32
    assert set(np.unique(train.y)).issubset(set(range(4)))
