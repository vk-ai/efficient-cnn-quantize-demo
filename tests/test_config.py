from efficient_cnn.config import load_config


def test_default_config_keys():
    cfg = load_config()
    assert "data" in cfg and "model" in cfg and "train" in cfg
    assert cfg["quantize"]["bits"] == 8
    assert 0.0 < cfg["prune"]["sparsity"] < 1.0
