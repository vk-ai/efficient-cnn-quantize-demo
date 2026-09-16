"""CPU-friendly tiny efficient CNN + quantization/pruning learning demo."""

from efficient_cnn.eval import run_before_after
from efficient_cnn.model import TinyEfficientCNN

__all__ = ["TinyEfficientCNN", "run_before_after"]
__version__ = "0.1.0"
