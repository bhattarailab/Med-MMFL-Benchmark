"""Unified command-line interface for  Med-MMFL: Multimodal Federated Learning Benchmark in Healthcare.

Provides a ``med-mmfl-bench`` CLI entry point for running federated
learning experiments. Installed via ``pip install -e .`` and exposed
through ``pyproject.toml``'s ``[project.scripts]`` section.

Usage::

    med-mmfl-bench --config configs/fedavg.yml --algorithm fedavg
    med-mmfl-bench --config configs/ehrxqa_scaffold.yml --algorithm scaffold \\
                 --comm-rounds 50 --name ehrxqa_exp

See Also:
    - ``python main.py`` for an equivalent standalone entry point.
    - ``configs/*.yml`` for example configurations.
"""

import argparse
import os
import sys
from importlib import import_module
from typing import Any, Optional

from med_mmfl_bench.utils.config import parse_config
from med_mmfl_bench.utils.load_dataset import load_dataset_with_splits
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.seed import set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Server resolution: maps config.dataset.dset_name → server subpackage
# ---------------------------------------------------------------------------
_SERVER_REGISTRY = {
    "ehrxqa": "med_mmfl_bench.servers.ehrxqa",
    "pathvqa": "med_mmfl_bench.servers.pathvqa",
    "brats24": "med_mmfl_bench.servers.brats",
    "symile_mimic": "med_mmfl_bench.servers.symilemimic",
    "mimic-cxr": "med_mmfl_bench.servers.mimiccxrjpg",
}

_ALGO_CLASS_MAP = {
    "fedavg": "FedavgServer",
    "fedprox": "FedproxServer",
    "scaffold": "ScaffoldServer",
    "fednova": "FednovaServer",
    "moon": "MoonServer",
    "creamfl": "CreamflServer",
}


def _create_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Med-MMFL: Multimodal Federated Learning Benchmark"
            "in Healthcare"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --config configs/fedavg.yml --algorithm fedavg\n"
            "  python main.py --config configs/ehrxqa_scaffold.yml "
            "--algorithm scaffold --comm-rounds 50\n"
        ),
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=list(_ALGO_CLASS_MAP.keys()),
        help="Federated learning algorithm.",
    )

    # Experiment settings
    parser.add_argument(
        "--comm-rounds",
        type=int,
        default=30,
        help="Number of communication rounds (default: 30).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Experiment name for logging (default: 'experiment').",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./experiments",
        help="Directory for experiment outputs (default: './experiments').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of client devices (default: 3).",
    )

    # WandB
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="med-mmfl-bench",
        help="WandB project name (default: 'med-mmfl-bench').",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name.",
    )

    return parser


def _init_wandb(args: argparse.Namespace) -> Optional[Any]:
    """Initialize Weights & Biases if enabled.

    Reads the API key from the ``WANDB_API_KEY`` environment variable.

    Returns:
        WandB run object, or None if disabled.
    """
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError:
        logger.warning(
            "wandb not installed. Install with: pip install wandb. "
            "Continuing without logging."
        )
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.info(
            "WANDB_API_KEY not set. Using cached credentials or anonymous mode."
        )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.name,
        config=vars(args),
    )
    return run


def _resolve_server_class(dset_name: str, algorithm: str) -> type:
    """Resolve the server class for a (dataset, algorithm) pair.

    Args:
        dset_name: Dataset name from ``config.dataset.dset_name``.
        algorithm: Algorithm name (e.g. ``"fedavg"``).

    Returns:
        The server class to instantiate.

    Raises:
        SystemExit: If the dataset or algorithm is not supported.
    """
    if dset_name not in _SERVER_REGISTRY:
        logger.error(
            "Unknown dataset '%s'. Supported: %s",
            dset_name,
            list(_SERVER_REGISTRY.keys()),
        )
        sys.exit(1)

    server_pkg = _SERVER_REGISTRY[dset_name]
    class_name = _ALGO_CLASS_MAP[algorithm]

    try:
        module = import_module(f"{server_pkg}.{algorithm}")
    except ModuleNotFoundError:
        logger.error(
            "Server module '%s.%s' not found. "
            "Check that '%s' is implemented for dataset '%s'.",
            server_pkg,
            algorithm,
            algorithm,
            dset_name,
        )
        sys.exit(1)

    if not hasattr(module, class_name):
        logger.error(
            "Class '%s' not found in module '%s'.",
            class_name,
            module.__name__,
        )
        sys.exit(1)

    return getattr(module, class_name)


def main(argv: Optional[list] = None) -> None:
    """Main entry point for the ``med-mmfl-bench`` CLI.

    Args:
        argv: Command-line arguments. If None, reads from ``sys.argv``.
    """
    parser = _create_parser()
    args = parser.parse_args(argv)

    # Reproducibility
    set_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # Load config
    config = parse_config(args.config)
    logger.info("Loaded config from: %s", args.config)
    logger.info(
        "Dataset: %s | Algorithm: %s",
        config.dataset.dset_name,
        args.algorithm,
    )

    # Create experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)

    # WandB
    wandb_run = _init_wandb(args)

    # Load datasets and partitions
    datasets = load_dataset_with_splits(config)
    logger.info(
        "Loaded %d client partitions.",
        len(datasets["client_datasets"]),
    )

    # Resolve and instantiate server
    server_class = _resolve_server_class(config.dataset.dset_name, args.algorithm)
    logger.info("Using server: %s", server_class.__name__)

    if config.dataset.dset_name == "mimic-cxr":
        logger.warning(
            "MIMIC-CXR uses a custom server that does not require "
            "validation/test datasets. Ignoring those fields."
        )

        server = server_class(
            args=args,
            config=config,
            wandb=wandb_run or False,
        )
    else:
        server = server_class(
            args=args,
            config=config,
            val_dataset=datasets["val_set"],
            test_dataset=datasets["test_set"],
            client_datasets=datasets["client_datasets"],
            wandb=wandb_run or False,
        )

    # Federated learning loop
    logger.info("=" * 60)
    logger.info("Starting federated training (%d rounds)", args.comm_rounds)
    logger.info("=" * 60)

    for round_idx in range(args.comm_rounds):
        logger.info(
            "--- Communication Round %d/%d ---",
            round_idx + 1,
            args.comm_rounds,
        )
        server.update()
    # server.evaluate()

    # Finalize
    server.finalize()
    logger.info("Federated training complete.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
