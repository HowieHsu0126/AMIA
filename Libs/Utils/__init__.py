"""Utility functions for data preprocessing, configuration management, etc."""

from importlib import import_module as _imp

__all__ = ["preprocessor", "config"]

globals().update({name: _imp(f"Libs.Utils.{name}") for name in __all__}) 