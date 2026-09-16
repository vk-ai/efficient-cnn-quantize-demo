import numpy as np

from efficient_cnn.data import train_test_split
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.train import train


def test_train_loss_drops():
    train_ds, _ = train_test_split(128, 32, 8, 8, 1, 4, 0.05, seed=0)
    rng = np.random.default_rng(0)
    m = TinyEfficientCNN(1, 8, 4, rng)
    hist = train(m, train_ds, epochs=12, lr=0.2, batch_size=32, seed=0)
    assert hist[-1] < hist[0] - 0.05
