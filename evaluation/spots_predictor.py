"""Spot predictor class."""

from typing import Union, Tuple, Dict

import numpy as np

import evaluation.util as util


def gauss_evaluate():
    """Use gauss fitting to evaluate."""
    return 0


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
        return gauss_evaluate()
