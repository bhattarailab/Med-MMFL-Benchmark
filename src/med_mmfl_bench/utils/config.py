"""Configuration parsing and serialization utilities.

Supports loading configs from YAML, JSON, pickle, and PyTorch files.
Config values are accessible as attributes via ``munch.Munch``.
"""

import os
import pickle
from typing import Any, Dict, Optional

import munch

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]

import yaml
from yaml.error import YAMLError

from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

__all__ = ["parse_config", "dump_config"]


def _loader(config_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Load configuration from a serialized file.

    Attempts to load as YAML first, then falls back to pickle and PyTorch
    formats.

    Args:
        config_path: Path to the configuration file.
        verbose: If True, log diagnostic messages on load failures.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        TypeError: If the file cannot be parsed by any supported loader.
        FileNotFoundError: If ``config_path`` does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fin:
        try:
            return yaml.safe_load(fin)
        except YAMLError:
            if verbose:
                logger.debug("Failed to load as YAML. Trying pickle loader.")

    with open(config_path, "rb") as fin:
        try:
            return pickle.load(fin)
        except (TypeError, pickle.UnpicklingError):
            if verbose:
                logger.debug("Failed to load as pickle. Trying torch loader.")
        if torch is not None:
            try:
                return torch.load(fin, weights_only=False)
            except Exception:
                if verbose:
                    logger.debug("Failed to load via torch.")

    raise TypeError(
        f"Cannot parse '{config_path}'. Supported formats: YAML, JSON, pickle, torch."
    )


def dump_config(
    config: munch.Munch,
    dump_to: str,
    overwrite: bool = False,
    serializer: str = "json",
) -> None:
    """Dump configuration to a local file.

    Args:
        config: Configuration object to serialize.
        dump_to: Destination file path.
        overwrite: If False, raises ``FileExistsError`` when ``dump_to`` exists.
        serializer: Output format. One of ``"json"``, ``"yaml"``, ``"pickle"``,
            ``"torch"``.

    Raises:
        ValueError: If ``serializer`` is not a supported format.
        FileExistsError: If ``dump_to`` exists and ``overwrite`` is False.
    """
    valid_serializers = {"json", "yaml", "pickle", "torch"}
    if serializer not in valid_serializers:
        raise ValueError(
            f"Serializer must be one of {valid_serializers}, got '{serializer}'"
        )

    if os.path.exists(dump_to) and not overwrite:
        raise FileExistsError(dump_to)

    mode = "wb" if serializer in ("pickle", "torch") else "w"
    config_dict = munch.unmunchify(config)

    with open(dump_to, mode) as fout:
        if serializer == "json":
            json.dump(config_dict, fout, indent=4, sort_keys=True)
        elif serializer == "yaml":
            yaml.dump(config_dict, fout)
        elif serializer == "pickle":
            pickle.dump(config_dict, fout)
        elif serializer == "torch":
            if torch is None:
                raise ImportError("PyTorch is required for torch serialization.")
            torch.save(config_dict, fout)


def parse_config(
    config_fname: str,
    delimiter: str = "__",
    strict_cast: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> munch.Munch:
    """Parse a configuration file with optional keyword overrides.

    Loads the configuration from ``config_fname`` and optionally overwrites
    values using ``kwargs``. Multi-depth keys use ``delimiter`` as separator
    (e.g., ``optimizer__learning_rate=0.001``).

    Args:
        config_fname: Path to the configuration file (YAML, JSON, pickle, or
            torch format).
        delimiter: Separator for nested key overrides in ``kwargs``.
        strict_cast: If True, cast overridden values to the type of the
            original config value.
        verbose: If True, log override operations.
        **kwargs: Key-value pairs to override in the configuration. Use
            ``delimiter`` for nested keys (e.g., ``train__lr=0.01``).

    Returns:
        Parsed configuration with attribute-style access.

    Raises:
        FileNotFoundError: If ``config_fname`` does not exist.
        KeyError: If a key in ``kwargs`` is not found in the config.

    Example:
        >>> config = parse_config("configs/fedavg.yml")
        >>> config.optimizer.learning_rate
        0.0001
        >>> config = parse_config("configs/fedavg.yml", optimizer__learning_rate=0.01)
        >>> config.optimizer.learning_rate
        0.01
    """
    config = _loader(config_fname, verbose)

    if kwargs and verbose:
        logger.info("Overwriting configurations: %s", kwargs)

    for arg_key, arg_val in kwargs.items():
        keys = arg_key.split(delimiter)
        n_keys = len(keys)

        _config = config
        for idx, _key in enumerate(keys):
            if _key not in _config:
                raise KeyError(
                    f"Config key '{arg_key}' not found. "
                    f"Available keys at depth {idx}: {list(_config.keys())}"
                )
            if n_keys - 1 == idx:
                if strict_cast:
                    typecast = type(_config[_key])
                    _config[_key] = typecast(arg_val)
                else:
                    _config[_key] = arg_val
            else:
                _config = _config[_key]

    return munch.munchify(config)
