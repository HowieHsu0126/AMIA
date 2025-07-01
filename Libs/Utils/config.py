import logging
from typing import Any, Dict, Union
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(cfg_path: Union[str, Path], *, safe: bool = True) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        cfg_path: Path-like object pointing to a ``.yml`` or ``.yaml`` file.
        safe:   Whether to use :func:`yaml.safe_load` (default) or the full
                loader.  ``safe=True`` is recommended unless you need advanced
                YAML tags.

    Returns:
        A nested ``dict`` that mirrors the YAML structure.

    Raises:
        FileNotFoundError: If *cfg_path* does not exist.
        yaml.YAMLError:    If the YAML file contains invalid syntax.

    Example:
        >>> cfg = load_config("config/train.yml")
        >>> batch_size = cfg["training"]["batch_size"]
    """
    cfg_path = Path(cfg_path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    loader = yaml.safe_load if safe else yaml.full_load

    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update mapping *d* with *u* (overrides win)."""
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = _deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _load_recursive(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            cfg_local = loader(fh) or {}

        # Handle optional "extends" field (string or list of str)
        base_field = cfg_local.pop("extends", None)
        if base_field:
            if isinstance(base_field, str):
                base_paths = [base_field]
            elif isinstance(base_field, list):
                base_paths = base_field
            else:
                raise TypeError("`extends` must be str or list of str")

            merged: Dict[str, Any] = {}
            for bp in base_paths:
                parent_cfg = _load_recursive((path.parent / bp).resolve())
                merged = _deep_update(merged, parent_cfg)
            cfg_local = _deep_update(merged, cfg_local)

        return cfg_local

    cfg = _load_recursive(cfg_path)

    logger.info("Loaded config (with inheritance): %s", cfg_path)
    return cfg 