"""MOON optimizer — FedAvg with model contrastive learning."""

from med_mmfl_bench.algorithms.fedavg import FedavgOptimizer

__all__ = ["MoonOptimizer"]


class MoonOptimizer(FedavgOptimizer):
    """MOON: Model-Contrastive Federated Learning.

    Uses the same aggregation strategy as FedAvg. The contrastive loss
    between global, local, and previous-round models is applied on the
    client side during local training.

    Reference:
        Li et al., "Model-Contrastive Federated Learning", CVPR 2021.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
