"""Util function for model processing."""
from typing import Tuple
import sys

import tensorflow as tf
import numpy as np
import skimage.util
import pandas as pd

sys.path.append("../")
from api.util import gauss_single_image, get_coordinate_list
from api.util_image import next_multiple_


PREDICTORS = ["Gauss localisation", "Network localisation"]


def predict_crop(
    crop: np.ndarray, model: tf.keras.models.Model, localisator: str = "Neural network"
) -> np.ndarray:
    """Predict on a crop of size needed for the network and return coordinates."""
    model_input_size = model.layers[0].output_shape[0][1]
    model_output_size = model.layers[-1].output_shape[1]

    if crop.shape[0] != model_input_size:
        raise ValueError(
            f"Model need input of shape [{model_input_size},{model_input_size}], image has shape {crop.shape}"
        )
    assert crop.shape[0] == crop.shape[1]

    model_cell_size = int(model_input_size / model.layers[-1].output_shape[1])

    pred = model.predict(crop[None, ..., None]).squeeze()

    if localisator == "Gaussian fitting":
        print("Using gauss 2D fitting for localisation!")
        pred = gauss_single_image(crop[..., None], pred, model_cell_size, model_cell_size).squeeze()

    coord = get_coordinate_list(pred, model_input_size, model_output_size)
    return coord[..., 0], coord[..., 1]


def predict_baseline(
    image: np.ndarray, model: tf.keras.models.Model, localisator: str = "Neural network"
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a binary or categorical model based prediction of an image.

    Args:
        - image: Image to be predicted.
        - model: Model used to predict the image.
        - localisator: use either Gauss or Network localisator
        - bit_depth: Bit depth to normalize images. Model dependent.

    Returns:
        - pred: list of coordinates [x,y].
    """
    model_input_size = model.layers[0].output_shape[0][1]

    # normalisation and padding
    image = image.copy() / np.max(image)
    pad_bottom = next_multiple_(image.shape[0], model_input_size) - image.shape[0]
    pad_right = next_multiple_(image.shape[1], model_input_size) - image.shape[1]
    image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "median")

    # predict on patches of the image and combine all the patches
    crops = skimage.util.view_as_windows(image, (model_input_size, model_input_size), step=model_input_size)
    all_coord_x = []
    all_coord_y = []
    for i in range(crops.shape[0]):
        for j in range(crops.shape[1]):
            x, y = predict_crop(crops[i, j], model, localisator)
            abs_coord_x = x + j * model_input_size
            abs_coord_y = y + i * model_input_size

            all_coord_x.append(abs_coord_x)
            all_coord_y.append(abs_coord_y)

    all_coord_x = np.concatenate(all_coord_x)
    all_coord_y = np.concatenate(all_coord_y)
    selection = (all_coord_x < image.shape[1]) & (all_coord_y < image.shape[0])
    all_coord_x = all_coord_x[selection]
    all_coord_y = all_coord_y[selection]

    return np.array(all_coord_x), np.array(all_coord_y)


def adaptive_prediction(
    images: np.ndarray, model: tf.keras.models.Model, localisator: str = "Neural network"
) -> pd.DataFrame:
    """Predicts images according to the selected model type.

    Args:
        - images (list of np.ndarray): List of images to be predicted.
        - model (tf.keras.models.Model): Model file.
        - localisator: use either Gauss or Network localisator

    Returns:
        - image (np.ndarray): Array containing the prediction.
    """
    pred = []
    index = 0
    for image in images:
        x, y = predict_baseline(image, model, localisator)
        pred.append([np.repeat(index, len(x)), x, y])
        index += 1

    pred_df = pd.DataFrame(np.concatenate(pred, axis=1).T, columns=["img_index", "x", "y"])
    return pred_df
