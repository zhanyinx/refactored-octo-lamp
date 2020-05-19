"""Util functions for image processing."""

import numpy as np
import skimage.io
import tifffile

ALLOWED_EXTENSIONS = ["tif", "tiff", "jpg", "jpeg", "stk", "png"]


def adaptive_imread(file: str) -> np.ndarray:
    """Opens images depending on their filetype."""
    if not any(file.endswith(i) for i in ALLOWED_EXTENSIONS):
        raise ValueError(f"File must end with {ALLOWED_EXTENSIONS}")

    return skimage.io.imread(file)


def __get_min_axis(image: np.ndarray) -> int:
    """Returns the index of a smallest axis of an image."""
    shape = image.shape
    axis = shape.index(min(shape))
    return axis


def adaptive_preprocessing(image: np.ndarray, image_type: str) -> np.ndarray:
    """Preprocesses images according to their selected image type."""
    image = image.squeeze()
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray but is {type(image)}.")
    if image.ndim not in [2, 3]:
        raise ValueError(f"image must be 2D or 3D but is {image.ndim}D.")

    axis = __get_min_axis(image)
    axis_len = image.shape[axis]

    if image_type == "One Frame (Grayscale or RGB)":
        return [skimage.color.rgb2gray(image)]
    if image_type == "Z-Stack":
        return [np.max(image, axis=axis)]
    if image_type == "All Frames":
        return [np.take(image, i, axis=axis) for i in range(axis_len)]
    if image_type == "Time-Series":
        return [np.take(image, i, axis=axis) for i in range(axis_len)]

    raise ValueError(f"Something went very wrong! Image type not available. Selected image type is {image_type}")


def next_multiple_(x: int, base: int = 512) -> int:
    """Calculates next mutiple of base given the input."""
    if x % base:
        x = x + (base - x % base)
    return x


# TO DO: save image + spot?
def adaptive_imsave(fname: str, image: np.ndarray, image_type: str) -> None:
    """Saves images according to their selected image type."""
    image = np.array(image).squeeze()

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be np.ndarray but is {type(image)}.")
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D but is {image.ndim}D.")

    if image_type in ["One Frame (Grayscale or RGB)", "Z-Stack", "Time-Series"]:
        skimage.io.imsave(fname, image)
    if image_type == "All Frames":
        tifffile.imsave(fname, image, imagej=True)
