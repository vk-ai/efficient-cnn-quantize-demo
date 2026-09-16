import numpy as np

from efficient_cnn.model import TinyEfficientCNN


def test_forward_shape():
    rng = np.random.default_rng(0)
    m = TinyEfficientCNN(1, 8, 4, rng)
    x = rng.random((5, 1, 8, 8))
    logits = m.forward(x)
    assert logits.shape == (5, 4)
    preds = m.predict(x)
    assert preds.shape == (5,)


def test_param_roundtrip():
    rng = np.random.default_rng(1)
    m = TinyEfficientCNN(1, 4, 3, rng)
    params = {k: v.copy() for k, v in m.named_params().items()}
    m2 = m.copy()
    for k in params:
        assert np.allclose(params[k], m2.named_params()[k])
