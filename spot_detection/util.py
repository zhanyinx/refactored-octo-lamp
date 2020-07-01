"""Utility helper functions."""

from typing import Callable
import importlib


def get_from_module(path: str, attribute: str) -> Callable:
    """Grabs an attribute (e.g. class) from a given module path."""
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute  # type: ignore[return-value]
