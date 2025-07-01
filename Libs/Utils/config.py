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
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = loader(f)

    logger.info("Loaded config: %s", cfg_path)
    return cfg 