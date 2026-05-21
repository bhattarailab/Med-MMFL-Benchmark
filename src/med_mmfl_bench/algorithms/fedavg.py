"""Federated Averaging (FedAvg) optimizer."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import copy

from med_mmfl_bench.algorithms.base import BaseOptimizer

__all__ = ["FedavgOptimizer"]


class FedavgOptimizer(BaseOptimizer):
    """FedAvg: Communication-Efficient Learning of Deep Networks.

    Implements the standard Federated Averaging algorithm where the server
    broadcasts its model to selected clients, each client trains locally,
    and the server aggregates the updated weights via weighted averaging.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
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
        omega = kwargs.get('omega', None)
        if not client_models:
            raise ValueError("No client models provided for aggregation.")

        # Average the parameters of the client models
        avg_weights = self.average_weights(client_models, omega)
        self.global_model.load_state_dict(avg_weights, strict=False)

        return avg_weights