"""SpotsModel class."""

from typing import Callable, Dict, Tuple

# import functools

import numpy as np
import tensorflow as tf

from spot_detection.datasets.dataset_spots import SpotsDataset
from spot_detection.datasets.dataset_sequence import DatasetSequence
from spot_detection.losses.f1_score import f1_score
from spot_detection.losses.L2_norm import l2_norm, f1_l2_combined_loss
from spot_detection.models.base import Model
from spot_detection.models.util import random_cropping, next_power
from spot_detection.networks.fcn import fcn, fcn_dropout, fcn_dropout_bigger
from spot_detection.networks.resnet import resnet

# from spot_detection.models.util_augment import augment_batch_baseline

DEFAULT_TRAIN_ARGS = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 3e-04,
}
DEFAULT_NETWORK_ARGS = {"n_classes": 3}
DEFAULT_LOSS = tf.keras.losses.binary_crossentropy
DEFAULT_OPTIMIZER = tf.keras.optimizers.Adam(DEFAULT_TRAIN_ARGS["learning_rate"])


class SpotsModel(Model):
    """ Model to predict spot localization. """

    def __init__(
        self,
        dataset_cls: type = SpotsDataset,
        network_fn: Callable = fcn,
        loss_fn: Callable = DEFAULT_LOSS,
        optimizer_fn: Callable = DEFAULT_OPTIMIZER,
        train_args: Dict = DEFAULT_TRAIN_ARGS,
        dataset_args: Dict = None,
        network_args: Dict = DEFAULT_NETWORK_ARGS,
        batch_format_fn: Callable = None,
        batch_augment_fn: Callable = None,
    ):
        # self.batch_augment_fn = functools.partial(
        #     augment_batch_baseline,
        #     flip_=self.dataset_args["flip"],
        #     illuminate_=self.dataset_args["illuminate"],
        #     gaussian_noise_=self.dataset_args["gaussian_noise"],
        #     rotate_=self.dataset_args["rotate"],
        #     translate_=self.dataset_args["translate"],
        #     cell_size=self.dataset_args["please also put this in the config"],
        # )
        super().__init__(
            dataset_cls,
            network_fn,
            loss_fn,
            optimizer_fn,
            train_args,
            dataset_args,
            network_args,
            batch_format_fn,
            batch_augment_fn,
        )

    @property
    def metrics(self) -> list:
        return ["accuracy", "mse", f1_score, l2_norm, f1_l2_combined_loss]

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """ Evaluates on a batch of images / masks. """
        return 0

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """ Predict on a single input image. """
        # if image.dtype == np.uint8:
        #     image = (image / 255).astype(np.float32)
        # if image.dtype == np.uint16:
        #     image = (image / 65535).astype(np.float32)

        # pad_bottom = next_power(image.shape[0]) - image.shape[0]
        # pad_right = next_power(image.shape[1]) - image.shape[1]
        # image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")
        pred = self.network.predict(image[None, ..., None], batch_size=1).squeeze()
        # pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]

        return pred
