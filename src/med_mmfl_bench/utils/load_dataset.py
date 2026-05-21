"""Dataset loading utilities for federated learning partitions.

Handles loading partition files and creating per-client dataset subsets
from full training datasets. Supports dynamic dataset resolution via
a registry or by accepting a dataset class directly.
"""

import pickle
from importlib import import_module
from typing import Any, Dict, List, Optional, Type

from torch.utils.data import Dataset, Subset

from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["load_dataset_with_splits"]

# Registry mapping dataset names to their module paths and class names.
# New datasets should be registered here for automatic resolution.
_DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "symile_mimic": {
        "module": "med_mmfl_bench.datasets.symile_mimic",
        "class": "SymileMIMICDataset",
    },
    "brats24": {
        "module": "med_mmfl_bench.datasets.brats",
        "class": "BraTS24GLIPostDataset",
    },
    "ehrxqa": {
        "module": "med_mmfl_bench.datasets.ehrxqa",
        "class": "EHRXQA",
    },
    "pathvqa": {
        "module": "med_mmfl_bench.datasets.pathvqa",
        "class": "YesNoVQADataset",
    },
    "mimic-cxr": {
        "module": "med_mmfl_bench.datasets.mimic_cxr",
        "class": "MimicMultiModal",
    },
}


def _resolve_dataset_class(dset_name: str) -> Type[Dataset]:
    """Resolve a dataset class by name from the registry.

    Args:
        dset_name: Dataset name matching a key in ``_DATASET_REGISTRY``.

    Returns:
        The resolved dataset class.

    Raises:
        KeyError: If ``dset_name`` is not in the registry.
    """
    if dset_name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dset_name}'. "
            f"Registered: {list(_DATASET_REGISTRY.keys())}"
        )

    entry = _DATASET_REGISTRY[dset_name]
    module = import_module(entry["module"])
    return getattr(module, entry["class"])


def load_dataset_with_splits(
    config: Any,
    partition_dir: str = "partitions",
) -> Dict[str, Any]:
    """Load a dataset and partition it into federated client splits.

    Reads a partition pickle file that defines train/val indices per client,
    and creates ``Subset`` objects for each client from the full dataset.

    The partition filename is constructed as::

        {partition_dir}/{dset_name}-{view}-{partition}[-client{clients}].pkl

    The ``-client{clients}`` suffix is appended only when
    ``config.dataset.clients`` is set.

    Args:
        config: Configuration object with ``dataset`` section containing:
            - ``dset_name``: Dataset identifier (must be in the registry).
            - ``view``: View/split identifier.
            - ``partition``: Partition strategy name.
            - ``img_path``: Root directory for images/data.
            - ``ann_path`` *(optional)*: Annotation directory.
            - ``clients`` *(optional)*: Number of clients (appended to
              partition filename if present).
        partition_dir: Directory containing partition pickle files.

    Returns:
        Dictionary with keys:
            - ``"train_set"``: Full training dataset.
            - ``"val_set"``: Full validation dataset.
            - ``"test_set"``: Full test dataset.
            - ``"client_datasets"``: List of dicts, each containing
              ``"client_id"``, ``"train_set"``, and ``"val_set"`` as
              ``Subset`` objects.

    Raises:
        FileNotFoundError: If the partition pickle file does not exist.
        KeyError: If the dataset name is not in the registry.

    Example:
        >>> config = parse_config("configs/symile_creamfl.yml")
        >>> datasets = load_dataset_with_splits(config)
        >>> len(datasets["client_datasets"])
        3
    """
    dset_name = config.dataset.dset_name
    dset_cfg = config.dataset

    # Build partition filename
    partition_stem = f"{dset_name}-{dset_cfg.view}-{dset_cfg.partition}"
    if hasattr(dset_cfg, "clients") and dset_cfg.clients is not None:
        partition_stem += f"-client{dset_cfg.clients}"
    partition_path = f"{partition_dir}/{partition_stem}.pkl"

    logger.info("Loading partition from: %s", partition_path)
    with open(partition_path, "rb") as f:
        data_partition = pickle.load(f)

    # Resolve dataset class from registry
    dataset_class = _resolve_dataset_class(dset_name)

    # Build constructor kwargs — different datasets accept different args
    ann_path = getattr(dset_cfg, "ann_path", None)
    common_kwargs: Dict[str, Any] = {}

    # Datasets that use (data_dir, ann_root, view_type, split) signature
    _POSITIONAL_DATASETS = {"ehrxqa", "pathvqa", "mimic-cxr"}
    if dset_name in _POSITIONAL_DATASETS:
        train_set = dataset_class(dset_cfg.img_path, ann_path, dset_cfg.view, split="train")
        val_set = dataset_class(dset_cfg.img_path, ann_path, dset_cfg.view, split="val")
        test_set = dataset_class(dset_cfg.img_path, ann_path, dset_cfg.view, split="test")
    else:
        # Keyword-based datasets (symile_mimic, brats24, etc.)
        common_kwargs = {
            "data_dir": dset_cfg.img_path,
            "ann_root": ann_path,
            "view_type": dset_cfg.view,
        }
        train_set = dataset_class(**common_kwargs, split="train")
        val_set = dataset_class(**common_kwargs, split="val")
        test_set = dataset_class(**common_kwargs, split="test")

    # Create per-client subsets
    client_partitions = data_partition["client"]
    client_datasets: List[Dict[str, Any]] = []

    for client_id, partition in client_partitions.items():
        train_idx = partition["train"]
        val_idx = partition["val"]
        client_datasets.append({
            "client_id": client_id,
            "train_set": Subset(train_set, train_idx),
            "val_set": Subset(train_set, val_idx),
        })

    logger.info(
        "Loaded %d client partitions for dataset '%s'",
        len(client_datasets),
        dset_name,
    )

    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "client_datasets": client_datasets,
    }