"""CPU-friendly tiny efficient CNN + quantization/pruning learning demo."""

from efficient_cnn.distill import run_kd_demo
from efficient_cnn.eval import run_before_after
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.qat import run_qat_demo

__all__ = ["TinyEfficientCNN", "run_before_after", "run_kd_demo", "run_qat_demo"]
__version__ = "0.1.0"
