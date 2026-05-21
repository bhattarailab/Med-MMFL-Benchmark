"""FedProx optimizer — FedAvg with proximal regularization."""

from med_mmfl_bench.algorithms.fedavg import FedavgOptimizer

__all__ = ["FedproxOptimizer"]


class FedproxOptimizer(FedavgOptimizer):
    """FedProx: Federated Optimization in Heterogeneous Networks.

    Uses the same aggregation as FedAvg. The proximal term is applied
    on the client side during local training, not during aggregation.

    Reference:
        Li et al., "Federated Optimization in Heterogeneous Networks",
        MLSys 2020.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
