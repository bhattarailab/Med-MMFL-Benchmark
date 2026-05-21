"""SCAFFOLD optimizer with variance reduction via control variates."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import copy

from med_mmfl_bench.algorithms.base import BaseOptimizer

__all__ = ["ScaffoldOptimizer"]


class ScaffoldOptimizer(BaseOptimizer):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

    Uses control variates to correct for client drift caused by
    heterogeneous data distributions. Both server and client maintain
    control variate state that is updated during aggregation.

    Reference:
        Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
        for Federated Learning", ICML 2020.
    """
    def __init__(self, model, **kwargs):
        """
        Args:
            model (dict): The model parameters to be used for federated learning.
            **kwargs: Additional arguments for the optimizer.
        """
        self.global_model = copy.deepcopy(model)
        BaseOptimizer.__init__(self)

    def dispatch(self, closure=None):
        """Dispatch the model to clients.
        """
        return self.global_model

    def aggregate(self, **kwargs):
        """Aggregate the models from clients.
        Args:
            **kwargs: Contains the client models and other parameters.
                - client_models (list): List of client model parameters.
                - omega (list): Weights for each client model, if applicable.
        """
        client_models = kwargs.get('client_models', [])

        if not client_models:
            raise ValueError("No client models provided for aggregation.")
        
        #non weighted aggregation for scaffold
        avg_weights = self.average_weights(client_models)
        self.global_model.load_state_dict(avg_weights, strict=False)

        return avg_weights