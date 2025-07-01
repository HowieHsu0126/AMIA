"""Top-level package for Neural Granger Causal Discovery (NGC-AKI).

This file turns the `Libs` directory into a proper Python package so that
modules can be imported as `import Libs.Models.cMLP` etc.  It also sets a
simple yet flexible logging configuration that can be overridden by the
parent application.
"""

import logging
import sys
from types import ModuleType

__all__ = [
    "Models",
    "Utils",
    "Data",
]

# ---------------------------------------------------------------------------
# Default logging setup ------------------------------------------------------
# ---------------------------------------------------------------------------
# Package-wide logger that libraries can reuse via ``logging.getLogger(__name__)``.
# This configuration will only take effect if the root logger has no handlers.
# If the user application configures logging first (recommended for production),
# the following block becomes a no-op.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    _root_logger.addHandler(_handler)
    _root_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Eager import sub-packages to avoid recursion issues ------------------------
# ---------------------------------------------------------------------------

import importlib as _imp


def _eager_import(name: str):
    """Import *name* and return the module object; guarantees real module."""
    module = _imp.import_module(name)
    sys.modules[name] = module
    return module


Models = _eager_import("Libs.Models")
Utils = _eager_import("Libs.Utils")
Data = _eager_import("Libs.Data")

__all__ = ["Models", "Utils", "Data"] 