from efficient_cnn.config import load_config
from efficient_cnn.eval import run_before_after


def test_eval_harness_keys_and_bounds():
    # Default config finishes in <1s on CPU and reliably learns the toy blobs.
    cfg = load_config()
    report = run_before_after(cfg)
    assert report["baseline"]["acc"] >= 0.70
    assert report["quantized"]["nbytes_int8_pack"] < report["baseline"]["nbytes_fp64"]
    assert report["pruned"]["nonzero"] < report["baseline"]["nonzero"]
    assert report["quantized"]["acc"] >= report["baseline"]["acc"] - 0.10
    assert report["pruned"]["acc"] >= report["baseline"]["acc"] - 0.40
    assert "flops" in report["baseline"]
    assert "prune_then_int8" in report
    assert report["prune_then_int8"]["order"] == ["prune", "quantize"]
