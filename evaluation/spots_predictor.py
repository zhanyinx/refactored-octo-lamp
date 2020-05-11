"""Spot predictor class."""

from typing import Union, Dict

import numpy as np

import evaluation.util as util
from spot_detection.losses.f1_score import f1_score
from spot_detection.losses.l2_norm import l2_norm


class SpotPredictor:
    """Given an image, detect spots."""
    def __init__(self, cfg: Dict):

        dataset_class_ = util.get_from_module("spot_detection.datasets", cfg["dataset"])
        model_class_ = util.get_from_module("spot_detection.models", cfg["model"])
        network_fn_ = util.get_from_module("spot_detection.networks", cfg["network"])
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
            image = util.read_image(image_or_filename)
        else:
            image = image_or_filename
        pred = self.model.predict_on_image(image)
        return pred

    def gauss_predict(self, image_or_filename: Union[np.ndarray, str]) -> np.ndarray:
        """Predict on a single image using gauss for localisation.

        If gauss cannot localise, it will use the model localisation.
        """
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename)
        else:
            image = image_or_filename

        image = image[None, ..., None]
        pred = self.model.network.predict(image)
        gauss_pred = util.gauss_single_image(image[0], pred[0], self.cell_size, self.crop_size)
        return gauss_pred

    def evaluate(self):
        """Evaluate on test part of a dataset."""
        if not self.use_gauss:
            return self.model.evaluate(self.dataset.x_test, self.dataset.y_test)

        x = self.dataset.x_test
        if x.ndim < 4:
            x = np.expand_dims(x, -1)
        preds = self.model.network.predict(x)

        gauss_preds = util.gauss_evaluate(x, preds, self.cell_size, self.crop_size)

        y_float32 = np.float32(self.dataset.y_test)
        gauss_preds = np.float32(gauss_preds)

        l2_norm_ = l2_norm(y_float32, gauss_preds) * self.cell_size
        f1_score_ = f1_score(y_float32, gauss_preds)

        return [f1_score_.numpy(), l2_norm_.numpy()]
