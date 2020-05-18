"""Util functions needed for other utils."""

import math
from typing import Tuple
import operator
import itertools

import numpy as np
import scipy.optimize as opt
import tensorflow as tf
import tensorflow.keras.backend as K


def get_prediction_matrix(spot_coord: np.ndarray, size: int, cell_size: int, size_y: int = None) -> np.ndarray:
    """Return np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Args:
        spot_coord: List of coordinates in x, y format with shape (n, 2).
        size: size of the image from which List of coordinates are extracted.
        cell_size: size of cell used to calculate F1 score, precision and recall.
        size_y: if not provided, it assumes it is squared image, otherwise the second shape of image

    Returns:
        - prediction (nd.nparray): numpy array of shape (n, n, 3): p, x, y format for each cell.
    """
    if not all(isinstance(i, int) for i in (size, cell_size)):
        raise TypeError(f"size and cell_size must be int, but are {type(size), type(cell_size)}.")

    nrow = math.ceil(size / cell_size)
    ncol = nrow
    if size_y is not None:
        ncol = math.ceil(size_y / cell_size)

    pred = np.zeros((nrow, ncol, 3))
    for nspot in range(len(spot_coord)):
        i = int(np.floor(spot_coord[nspot, 0])) // cell_size
        j = int(np.floor(spot_coord[nspot, 1])) // cell_size
        rel_x = (spot_coord[nspot, 0] - i * cell_size) / cell_size
        rel_y = (spot_coord[nspot, 1] - j * cell_size) / cell_size
        pred[i, j, 0] = 1
        pred[i, j, 1] = rel_x
        pred[i, j, 2] = rel_y

    return pred


def get_coordinate_list(matrix: np.ndarray, size_image: int = 512, size_grid: int = 64) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    Note - if plotting with plt.scatter, x and y must be reversed!

    Args:
        matrix: Matrix representation of spot coordinates.
        size_image: Default image size the grid was layed on.
        size_grid: Number of grid cells used.

    Returns:
        Array of x, y coordinates with the shape (n, 2).
    """
    size_gridcell = size_image // size_grid
    coords_x = []
    coords_y = []

    # Top left coordinates of every cell
    grid = np.array([x * size_gridcell for x in range(size_grid)])

    # TODO use np.where instead.
    for x, y in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[x, y, 0] > 0.5:
            grid_x = grid[x]
            grid_y = grid[y]
            spot_x = matrix[x, y, 1]
            spot_y = matrix[x, y, 2]

            coord_abs = get_absolute_coordinates(
                coord_spot=(spot_x, spot_y), coord_cell=(grid_x, grid_y), size_gridcell=size_gridcell,
            )

            coords_x.append(coord_abs[0])
            coords_y.append(coord_abs[1])

    return np.array([coords_y, coords_x]).T


def get_absolute_coordinates(
    coord_spot: Tuple[np.float32, np.float32], coord_cell: Tuple[np.float32, np.float32], size_gridcell: int = 8
) -> Tuple[np.float32, np.float32]:
    """Return the absolute image coordinates from relative cell coordinates."""
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_rel = tuple(map(lambda x: x * size_gridcell, coord_spot))
    coord_abs = tuple(map(operator.add, coord_cell, coord_rel))
    # coord_abs = tuple(map(lambda x: int(x), coord_abs))
    return coord_abs  # type: ignore


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
        return x_coord, y_coord

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1

    # if predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return x_coord, y_coord

    # return x0, y0
    return x_coord, y_coord


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


def recall_score(y_true, y_pred):
    """Recall score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    """F1 = 2 * (precision*recall) / (precision+recall)."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1_score_loss(y_true, y_pred):
    """F1 score loss."""
    return 1 - f1_score(y_true, y_pred)


def l2_norm(y_true, y_pred):
    """Calculate L2 norm between true and predicted coordinates."""
    coord_true = y_true[..., 1:]
    coord_pred = y_pred[..., 1:]

    comparison = tf.equal(coord_true, tf.constant(0, dtype=tf.float32))

    coord_true_new = tf.where(comparison, tf.zeros_like(coord_true), coord_true)
    coord_pred_new = tf.where(comparison, tf.zeros_like(coord_pred), coord_pred)

    l2_norm_ = K.sqrt(K.mean(K.sum(K.square(coord_true_new - coord_pred_new), axis=-1)))

    return l2_norm_


def f1_l2_combined_loss(y_true, y_pred):
    """Combined loss."""
    return l2_norm(y_true, y_pred) + f1_score_loss(y_true, y_pred)
