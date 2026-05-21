"""CreamFL optimizer — FedAvg with cross-modal knowledge distillation."""

from med_mmfl_bench.algorithms.fedavg import FedavgOptimizer

__all__ = ["CreamflOptimizer"]


class CreamflOptimizer(FedavgOptimizer):
    """CreamFL: Multimodal Federated Learning with Cross-Modal Distillation.

    Uses the same base aggregation strategy as FedAvg. Cross-modal
    knowledge distillation and inter/intra-modal alignment are handled
    on the client and server sides during training.

    Reference:
        Yu et al., "Multimodal Federated Learning via Contrastive
        Representation Ensemble", ICLR 2023.
    """

    def __init__(self, model, **kwargs):
        super(CreamflOptimizer, self).__init__(model=model, **kwargs)
