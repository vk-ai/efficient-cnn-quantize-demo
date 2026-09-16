import numpy as np

from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.quantize import fake_quantize_tensor, quantize_model, _affine_params


def test_fake_quant_roundtrip_close():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.5, size=(16, 8))
    scale, zp = _affine_params(w, 8)
    wq = fake_quantize_tensor(w, scale, zp, 8)
    # Max error roughly half an LSB * scale
    assert np.max(np.abs(w - wq)) <= scale * 0.5 + 1e-9


def test_quantize_shrinks_pack_size():
    rng = np.random.default_rng(2)
    m = TinyEfficientCNN(1, 8, 4, rng)
    q, stats = quantize_model(m, bits=8)
    assert stats.nbytes_int8 < stats.nbytes_fp64
    loss, acc = q.loss_acc(rng.random((8, 1, 8, 8)), rng.integers(0, 4, size=8))
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0
