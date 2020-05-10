"""Utility functions for spot detection module."""

import os
import importlib
from typing import Callable

import numpy as np
import skimage.io


def read_image(image_uri: str) -> np.ndarray:
    """Read image_uri."""
    def read_image_from_filename(image_filename):
        return skimage.io.imread(image_filename)

    local_file = os.path.exists(image_uri)

    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri)
        assert img is not None
    except ValueError:
        raise ValueError(f"Could not load image at {image_uri}")
    return img


def get_from_module(path: str, attribute: str) -> Callable:
    """Grabs an attribute from a given module path."""
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute  # type: ignore[return-value]
