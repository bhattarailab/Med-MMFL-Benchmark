"""FedNova optimizer with normalized averaging."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import copy

from med_mmfl_bench.algorithms.fedavg import FedavgOptimizer

__all__ = ["FednovaOptimizer"]


class FednovaOptimizer(FedavgOptimizer):
    """FedNova: Tackling the Objective Inconsistency Problem in FL.

    Normalizes gradients by the number of local update steps to mitigate
    objective inconsistency caused by heterogeneous local computations.

    Reference:
        Wang et al., "Tackling the Objective Inconsistency Problem in
        Heterogeneous Federated Optimization", NeurIPS 2020.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)


    def aggregate(self, **kwargs):
        """Aggregate the models from clients.
        Args:
            **kwargs: Contains the client models and other parameters.
                - client_models (list): List of client model parameters.
                - omega (list): Weights for each client model, if applicable.
        """
        pis = kwargs.get('pis', [])
        dis = kwargs.get('dis', None)
        ais = kwargs.get('ais', None)

        if not ais:
            raise ValueError("No client models provided for aggregation.")

        # Average the parameters of the client models
        sum_pis = sum(pis)
        coeff = np.dot(pis, ais) / sum_pis
        print("coeff: ", coeff)

        d_avg = copy.deepcopy(self.global_model.state_dict())
        
        for key in dis[0].keys():
            avg_molecule = 0
            # if 'weight' in key or 'bias' in key:
            for i in range(len(dis)):
                if i == 0:
                    avg_molecule = dis[i][key] * pis[i] / sum_pis
                else:
                    avg_molecule += dis[i][key] * pis[i] / sum_pis
            d_avg[key] = coeff * copy.deepcopy(avg_molecule)

        return d_avg