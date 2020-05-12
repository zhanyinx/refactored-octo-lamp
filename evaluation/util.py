"""Utility functions for evaluation module."""

import os
import importlib
from typing import Callable

import numpy as np
import skimage.io
import scipy.optimize as opt

from training.util_prepare import get_prediction_matrix
from training.util_prediction import get_coordinate_list


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


def gauss_2d(xy, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(((x - x0) ** (2) / (2 * sigma_xy ** (2))) + ((y - y0) ** (2) / (2 * sigma_xy ** (2))))
    )
    return gauss


def gauss_single_spot(image: np.ndarray, x_coord: float, y_coord: float, crop_size: int):
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max([int(np.round(y_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(x_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    assert end_dim2 - start_dim2 == crop_size
    assert end_dim1 - start_dim1 == crop_size

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Middle of the crop
    y0 = int(crop.shape[1] // 2)  # Middle of the crop
    sigma = max(*crop.shape) * 0.1  # 10% of the crop
    amplitude_max = np.max(crop) / 2  # Maximum value of the crop
    initial_guess = [amplitude_max, x0, y0, sigma, 0]

    lower = [0, 0, 0, 0, 0]
    upper = [np.max(crop), crop.shape[0], crop.shape[1], np.inf, np.max(crop)]
    bounds = [lower, upper]

    try:
        popt, _ = opt.curve_fit(gauss_2d, (xx.ravel(), yy.ravel()), crop.ravel(), p0=initial_guess, bounds=bounds)
    except RuntimeError:
        return y_coord, x_coord

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1

    # if predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return y_coord, x_coord

    return x0, y0


def gauss_single_image(image: np.ndarray, mask: np.ndarray, cell_size: int = 4, crop_size: int = 4):
    """Gaussian prediction on a single image."""
    prediction_coord = []
    coord_list = get_coordinate_list(mask, image.shape[0], image.shape[0] // cell_size)
    for i in range(len(coord_list)):
        x_coord = coord_list[i, 0]
        y_coord = coord_list[i, 1]

        # Avoid spots at the border of the image (out of the grid in the pred np.ndarray)
        if x_coord >= len(image):
            x_coord = len(image) - 0.0001
        if y_coord == len(image):
            y_coord = len(image) - 0.0001

        prediction_coord.append(gauss_single_spot(image, x_coord, y_coord, crop_size))

    if not prediction_coord:
        return mask
    # prediction_coord = [x for x in prediction_coord if any(v != 0 for v in x)]
    prediction_coord = np.flip(np.array(prediction_coord), axis=1)
    pred = get_prediction_matrix(prediction_coord, image.shape[0], cell_size, image.shape[1])
    return np.array(pred)


def gauss_evaluate(images: np.ndarray, masks: np.ndarray, cell_size: int = 4, crop_size: int = 4):
    """Use gauss fitting to calculate coordinates."""
    gauss_preds = []
    if images.shape[-1] == 1:
        for i in np.arange(images.shape[0]):
            pred = gauss_single_image(images[i], masks[i], cell_size, crop_size)
            gauss_preds.append(pred)
    return np.array(gauss_preds)
