"""Fed-Multimodal-Benchmark: A benchmark for federated learning on multimodal medical data.

This package provides implementations of popular federated learning algorithms
(FedAvg, FedProx, SCAFFOLD, MOON, CreamFL) applied to multimodal medical
datasets including SYMILE-MIMIC (CXR + ECG + Labs) and MIMIC-CXR (Image + Text).

Example:
    >>> from fed_mm_bench.utils.config import parse_config
    >>> config = parse_config("configs/fedavg.yml")
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
