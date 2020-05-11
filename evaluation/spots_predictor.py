"""Spot predictor class."""

from typing import Union, Tuple, Dict

import numpy as np
import scipy.optimize as opt

import evaluation.util as util
from training.util_prepare import get_prediction_matrix
from training.util_prediction import get_coordinate_list
from spot_detection.losses.f1_score import f1_score
from spot_detection.losses.l2_norm import l2_norm


def gauss_2d(xy, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(((x - x0) ** (2) / (2 * sigma_xy ** (2))) + ((y - y0) ** (2) / (2 * sigma_xy ** (2))))
    )
    return gauss


def gauss_single_spot(image: np.ndarray, x_coord: float, y_coord: float, cell_size: int):
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max(int(np.round(y_coord - cell_size // 2)), 0)
    end_dim1 = np.min([start_dim1 + cell_size, len(image) - 1])

    start_dim2 = np.max(int(np.round(x_coord - cell_size // 2)), 0)
    end_dim2 = np.min([start_dim2 + cell_size, len(image) - 1])
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
        popt, pcov = opt.curve_fit(gauss_2d, (xx.ravel(), yy.ravel()), crop.ravel(), p0=initial_guess, bounds=bounds)
    except RuntimeError:
        return 0, 0

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1

    return x0, y0


def gauss_single_image(image: np.ndarray, mask: np.ndarray, cell_size: int):
    """Gaussian prediction on a single image."""
    prediction_coord = []
    coord_list = get_coordinate_list(mask, image.shape[0], image.shape[0] // cell_size)
    for i in range(len(coord_list)):
        x_coord = coord_list[i, 0]
        y_coord = coord_list[i, 1]
        prediction_coord.append(gauss_single_spot(image, x_coord, y_coord, cell_size))

    prediction_coord = [x for x in prediction_coord if any(v != 0 for v in x)]
    return np.array(prediction_coord)


def gauss_evaluate(images: np.ndarray, masks: np.ndarray, cell_size: int = 4):
    """Use gauss fitting to calculate coordinates."""
    gauss_preds = []
    if images.shape[-1] == 1:
        for i in np.arange(images.shape[0]):
            xy = gauss_single_image(images[i], masks[i], cell_size)
            pred = get_prediction_matrix(np.flip(xy, axis=1), images.shape[1], cell_size, images.shape[2])
            gauss_preds.append(pred)
    return np.array(gauss_preds)


class SpotPredictor:
    """Given an image of a single handwritten character, recognizes it."""

    def __init__(self, cfg: Dict):

        dataset_class_ = util.get_from_module("spot_detection.datasets", cfg["dataset"])
        model_class_ = util.get_from_module("spot_detection.models", cfg["model"])
        network_fn_ = util.get_from_module("spot_detection.networks", cfg["network"])
        optimizer_fn_ = None
        loss_fn_ = None

        network_args = cfg.get("network_args", {})
        dataset_args = cfg.get("dataset_args", {})
        train_args = cfg.get("train_args", {})

        self.cell_size = dataset_args["cell_size"]
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

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, use_model: bool = True):
        """Evaluate on a dataset."""
        if use_model:
            return self.model.evaluate(self.dataset.x_test, self.dataset.y_test)

        x = self.dataset.x_test
        if x.ndim < 4:
            x = np.expand_dims(x, -1)
        preds = self.model.network.predict(x)

        gauss_preds = gauss_evaluate(x, preds, self.cell_size)
        y_float32 = np.float32(self.dataset.y_test)
        l2_norm_ = l2_norm(y_float32, gauss_preds) * self.cell_size
        f1_score_ = f1_score(y_float32, gauss_preds)

        return [f1_score_.numpy(), l2_norm_.numpy()]
