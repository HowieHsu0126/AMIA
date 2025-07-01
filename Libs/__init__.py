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
# Convenience sub-package imports (optional) --------------------------------
# ---------------------------------------------------------------------------
# Lazily expose sub-modules so that they can be accessed as attributes, e.g.:
# >>> import Libs
# >>> Libs.Models.cMLP
# This avoids importing heavy dependencies at package import time.
# ---------------------------------------------------------------------------

import importlib

class _LazyModule(ModuleType):
    """Lazy loader for sub-packages to keep import times minimal."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__dict__["_module_name"] = name

    def _load(self):
        module = importlib.import_module(self.__dict__["_module_name"])
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

# Expose sub-packages lazily
import sys as _sys
for _sub in ("Libs.Models", "Libs.Utils", "Libs.Data"):
    _sys.modules[_sub] = _LazyModule(_sub) 