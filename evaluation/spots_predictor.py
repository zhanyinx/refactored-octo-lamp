"""Spot predictor class."""

from typing import Union, Dict
import sys
import os

import numpy as np
import skimage.io
import scipy.optimize as opt

sys.path.append('../')

from spot_detection.data import get_prediction_matrix, get_coordinate_list
from spot_detection.losses import f1_score, l2_norm
from spot_detection.util import get_from_module


def read_image(image_uri: str) -> np.ndarray:
    """Read image_uri."""
    local_file = os.path.exists(image_uri)

    try:
        img = None
        if local_file:
            img = skimage.io.imread(image_uri)
            img = img / np.max(img)
        assert img is not None
    except ValueError:
        raise ValueError(f"Could not load image at {image_uri}")
    return img


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


class SpotPredictor:
    """Given an image, detect spots."""

    def __init__(self, cfg: Dict):

        dataset_class_ = get_from_module("spot_detection.datasets", cfg["dataset"])
        model_class_ = get_from_module("spot_detection.models", cfg["model"])
        network_fn_ = get_from_module("spot_detection.networks", cfg["network"])
        optimizer_fn_ = None
        loss_fn_ = None

        network_args = cfg.get("network_args", {})
        dataset_args = cfg.get("dataset_args", {})
        train_args = cfg.get("train_args", {})
        pred_args = cfg.get("pred_args", {})

        self.cell_size = dataset_args["cell_size"]
        self.use_gauss = pred_args["use_gauss"]
        self.dataset = dataset_class_(dataset_args["version"])
        self.dataset.load_data()

        self.model = model_class_(
            dataset_args=dataset_args,
            dataset_cls=self.dataset,
            loss_fn=loss_fn_,
            network_args=network_args,
            network_fn=network_fn_,
            optimizer_fn=optimizer_fn_,
            train_args=train_args,
        )
        self.model.load_weights()

        if self.use_gauss:
            self.crop_size = pred_args["crop_size"]

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> np.ndarray:
        """Predict on a single image."""
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename)
        else:
            image = image_or_filename
        pred = self.model.predict_on_image(image)
        return pred

    def gauss_predict(self, image_or_filename: Union[np.ndarray, str]) -> np.ndarray:
        """Predict on a single image using gauss for localisation.

        If gauss cannot localise, it will use the model localisation.
        """
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename)
        else:
            image = image_or_filename

        image = image[None, ..., None]
        pred = self.model.network.predict(image)
        gauss_pred = gauss_single_image(image[0], pred[0], self.cell_size, self.crop_size)
        return gauss_pred

    def evaluate(self):
        """Evaluate on test part of a dataset."""
        if not self.use_gauss:
            return self.model.evaluate(self.dataset.x_test, self.dataset.y_test)

        x = self.dataset.x_test
        if x.ndim < 4:
            x = np.expand_dims(x, -1)
        preds = self.model.network.predict(x)

        gauss_preds = gauss_evaluate(x, preds, self.cell_size, self.crop_size)

        y_float32 = np.float32(self.dataset.y_test)
        gauss_preds = np.float32(gauss_preds)

        l2_norm_ = l2_norm(y_float32, gauss_preds) * self.cell_size
        f1_score_ = f1_score(y_float32, gauss_preds)

        return [f1_score_.numpy(), l2_norm_.numpy()]
