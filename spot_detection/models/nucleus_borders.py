import numpy as np
import tensorflow as tf

from typing import Callable, Dict

from image_segmentation.datasets.dataset_sequence import DatasetSequence
from image_segmentation.datasets.nucleus_dataset import NucleusDataset
from image_segmentation.losses.dice_coefficient import dice_coef
from image_segmentation.models.base import Model
from image_segmentation.models.util import random_cropping, next_power, add_complete_borders
from image_segmentation.networks.fcn import fcn32

DEFAULT_TRAIN_ARGS = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 3e-04,
}
DEFAULT_NETWORK_ARGS = {
    "n_classes": 3
}
DEFAULT_LOSS = tf.keras.losses.binary_crossentropy
DEFAULT_OPTIMIZER = tf.keras.optimizers.Adam(
    DEFAULT_TRAIN_ARGS["learning_rate"])


def format_batch(images: np.ndarray, masks: np.ndarray):
    """ Formatting function for training and validation. """
    crop_images = []
    crop_masks = []

    for image, mask in zip(images, masks):
        crop_image, crop_mask = random_cropping(
            image=image, mask=mask, crop_size=256)
        crop_mask = add_complete_borders(mask=crop_mask, border_size=2)
        crop_images.append(crop_image)
        crop_masks.append(crop_mask)

    crop_images = np.array(crop_images, dtype=np.float32)
    crop_masks = np.array(crop_masks, dtype=np.float32)

    return crop_images, crop_masks


class NucleusModelBorders(Model):
    """ Nuclear Model to predict nuclei (semantic). """

    def __init__(
        self,
        dataset_cls: type = NucleusDataset,
        network_fn: Callable = fcn32,
        loss_fn: Callable = DEFAULT_LOSS,
        optimizer_fn: Callable = DEFAULT_OPTIMIZER,
        train_args: Dict = DEFAULT_TRAIN_ARGS,
        dataset_args: Dict = None,
        network_args: Dict = DEFAULT_NETWORK_ARGS,
    ):
        super().__init__(dataset_cls, network_fn, loss_fn,
                         optimizer_fn, train_args, dataset_args, network_args)

        self.batch_format_fn = format_batch
        self.batch_augment_fn = None

        self.network.summary()


    @property
    def metrics(self) -> list:
        return ["accuracy", dice_coef]

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """ Evaluates on a batch of images / masks. """
        scores = []
        for image, mask in zip(x, y):
            image = image.squeeze()
            mask = add_complete_borders(mask.squeeze())

            pad_bottom = next_power(image.shape[0]) - image.shape[0]
            pad_right = next_power(image.shape[1]) - image.shape[1]
            image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")
            pred = self.network.predict(image[None, ..., None]).squeeze()
            pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]

            score = np.mean(np.square(pred) - np.square(mask))
            scores.append(score)

        return np.mean(scores)

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """ Predict on a single input image. """
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        if image.dtype == np.uint16:
            image = (image / 65535).astype(np.float32)

        pad_bottom = next_power(image.shape[0]) - image.shape[0]
        pad_right = next_power(image.shape[1]) - image.shape[1]
        image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")
        pred = self.network.predict(
            image[None, ..., None], batch_size=1).squeeze()
        pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]

        return pred
