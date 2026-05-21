"""Federated optimization algorithm implementations.

Provides weight aggregation strategies for different federated learning
algorithms: FedAvg, FedProx, SCAFFOLD, FedNova, MOON, CreamFL.
"""

from med_mmfl_bench.algorithms.base import BaseOptimizer
from med_mmfl_bench.algorithms.fedavg import FedavgOptimizer
from med_mmfl_bench.algorithms.fedprox import FedproxOptimizer
from med_mmfl_bench.algorithms.scaffold import ScaffoldOptimizer
from med_mmfl_bench.algorithms.fednova import FednovaOptimizer
from med_mmfl_bench.algorithms.moon import MoonOptimizer
from med_mmfl_bench.algorithms.creamfl import CreamflOptimizer

__all__ = [
    "BaseOptimizer",
    "FedavgOptimizer",
    "FedproxOptimizer",
    "ScaffoldOptimizer",
    "FednovaOptimizer",
    "MoonOptimizer",
    "CreamflOptimizer",
]
