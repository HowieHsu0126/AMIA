"""Subpackage containing all Neural-GC backbone architectures.

Exposes:
    cMLP
    cRNN
    cLSTM
"""

from importlib import import_module as _imp

# Lazy-import heavy modules only when accessed ------------------------------------------------
__all__ = ["cMLP", "cRNN", "cLSTM"]

globals().update({name: _imp(f"Libs.Models.{name}") for name in __all__}) 