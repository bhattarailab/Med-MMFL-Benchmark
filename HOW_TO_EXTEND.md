# How to Extend Med-MMFL: Adding New Algorithms

This guide explains how to add new federated learning algorithms to the Med-MMFL framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Interfaces](#core-interfaces)
3. [Registry and Dynamic Resolution](#registry-and-dynamic-resolution)
4. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
5. [Checklist for New Algorithms](#checklist-for-new-algorithms)

---

## Overview

Med-MMFL uses a **plug-and-play architecture** where new algorithms can be added without modifying the core training loops. The framework utilizes dynamic module imports at runtime to find and instantiate algorithm-specific components:

- **Optimizer (`BaseOptimizer`)**: Implements server-side weight aggregation and parameter dispatch logic.
- **Client (`BaseClient`)**: Implements client-side local training (`update`), weight downloading (`download`), and parameter uploading (`upload`).
- **Server (`BaseServer` / `FedavgServer`)**: Orchestrates the communication rounds by dispatching, triggering local training, and calling the optimizer to aggregate weights.

---

## Core Interfaces

### 1. BaseOptimizer (Server-Side Optimization)

Defined in [base.py](file:///Users/aavashchhetri/Desktop/workspace/bbmmll/benchmark/fed-multimodal-benchmark/src/med_mmfl_bench/algorithms/base.py):

```python
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import torch

class BaseOptimizer(metaclass=ABCMeta):
    """Abstract base class for federated aggregation strategies."""
    
    @abstractmethod
    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch server model parameters.
        
        Typically returns the global model or a state dictionary.
        """
        raise NotImplementedError
    
    @abstractmethod
    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        """Aggregate client parameters and update the global model.
        
        Typically accepts client model state dicts and sample counts.
        """
        raise NotImplementedError
    
    @staticmethod
    def average_weights(
        w: List[Dict[str, torch.Tensor]],
        omega: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of model state dicts.
        
        Args:
            w: List of client model state dicts.
            omega: Optional weights for each client (default: uniform).
        
        Returns:
            Averaged model state dict.
        """
        if omega is None:
            omega = np.ones(len(w)) / len(w)
        else:
            omega = omega / omega.sum()
        
        w_avg = {}
        for key in w[0].keys():
            w_avg[key] = sum(omega[i] * w[i][key].float() for i in range(len(w)))
        return w_avg
```

### 2. BaseClient (Client-Side Training)

Defined in [base.py](src/med_mmfl_bench/clients/base.py):

```python
from abc import ABCMeta, abstractmethod
from typing import Any

class BaseClient(metaclass=ABCMeta):
    """Abstract base class for federated learning clients."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Perform local training on private data."""
        raise NotImplementedError

    @abstractmethod
    def upload(self) -> Any:
        """Upload local model parameters to the server.

        Returns:
            State dictionary or algorithm-specific upload payload.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, model: Any) -> None:
        """Download the global model from the server.

        Args:
            model: Global model object or state dictionary.
        """
        raise NotImplementedError
```

---

## Registry and Dynamic Resolution

When running an experiment (e.g., via `--algorithm customfl`), Med-MMFL dynamically imports and resolves classes:

1. **Server Resolution**: In `src/med_mmfl_bench/cli.py`, the dataset name maps to a package (e.g., `"brats24": "med_mmfl_bench.servers.brats"`), and the algorithm maps to a class name (e.g., `"customfl": "CustomflServer"`).
2. **Algorithm Optimizer**: The server imports `med_mmfl_bench.algorithms.customfl.CustomflOptimizer`.
3. **Client Class**: The server imports `med_mmfl_bench.clients.{dataset_name}.customfl.CustomflClient`.

---

## Step-by-Step Implementation Guide

To add a new algorithm named `customfl`:

### Step 1: Implement the Optimizer
Create `src/med_mmfl_bench/algorithms/customfl.py` containing `CustomflOptimizer` inheriting from `BaseOptimizer`.

### Step 2: Register the Optimizer
Add the optimizer to `src/med_mmfl_bench/algorithms/__init__.py`.

### Step 3: Implement Dataset-Specific Clients
For each dataset (e.g. `brats`), implement `CustomflClient` inheriting from `FedavgClient` or `BaseClient` in `src/med_mmfl_bench/clients/{dataset_name}/customfl.py`.

### Step 4: Implement Dataset-Specific Servers
Implement `CustomflServer` inheriting from `FedavgServer` or `BaseServer` in `src/med_mmfl_bench/servers/{dataset_name}/customfl.py`.

### Step 5: Register the Algorithm in CLI
Add the server class name mapping (e.g. `"customfl": "CustomflServer"`) to `_ALGO_CLASS_MAP` in `src/med_mmfl_bench/cli.py`.

---

## Checklist for New Algorithms

- [ ] Implement `CustomflOptimizer` in `src/med_mmfl_bench/algorithms/customfl.py`
- [ ] Add `CustomflOptimizer` to `src/med_mmfl_bench/algorithms/__init__.py`
- [ ] Implement `CustomflClient` in `src/med_mmfl_bench/clients/{dataset}/customfl.py` for all desired datasets (inheriting from `FedavgClient`)
- [ ] Implement `CustomflServer` in `src/med_mmfl_bench/servers/{dataset}/customfl.py` for all desired datasets (inheriting from `FedavgServer`)
- [ ] Add the algorithm name and server class mapping to `_ALGO_CLASS_MAP` in `src/med_mmfl_bench/cli.py`
- [ ] Create a YAML configuration file under `configs/customfl.yml`
- [ ] Run experiment with `med-mmfl-bench --config configs/customfl.yml --algorithm customfl`
