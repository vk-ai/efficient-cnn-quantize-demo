import numpy as np

from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.prune import magnitude_prune_tensor, prune_model


def test_magnitude_prune_sparsity():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(100,))
    out = magnitude_prune_tensor(w, 0.5)
    assert np.count_nonzero(out) == 50


def test_prune_model_reduces_nnz():
    rng = np.random.default_rng(3)
    m = TinyEfficientCNN(1, 8, 4, rng)
    before = sum(np.count_nonzero(v) for k, v in m.named_params().items() if k.endswith("_w"))
    p, stats = prune_model(m, 0.5)
    after = sum(np.count_nonzero(v) for k, v in p.named_params().items() if k.endswith("_w"))
    assert after < before
    assert stats.pruned > 0
