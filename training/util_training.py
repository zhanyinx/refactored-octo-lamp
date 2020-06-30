"""Training functions."""

import importlib
import platform
import sys
import time
from typing import Dict, Callable

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb

sys.path.append("../")

from spot_detection.datasets import Dataset
from spot_detection.models import Model
from util_prediction import get_coordinate_list

DEFAULT_CELL_SIZE = 4


def get_from_module(path: str, attribute: str) -> Callable:
    """Grabs an attribute from a given module path."""
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute  # type: ignore[return-value]


class WandbImageLogger(tf.keras.callbacks.Callback):
    """Custom image prediction logger callback in wandb.

    Expects segmentation images and the model class to have a predict_on_image method.
    """
    def __init__(
        self, model_wrapper: Model, dataset: Dataset, cell_size: int = DEFAULT_CELL_SIZE, example_count: int = 4
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.valid_images = dataset.x_valid[:example_count]  # type: ignore[index]
        self.train_images = dataset.x_train[:example_count]  # type: ignore[index]
        self.train_masks = dataset.y_train[:example_count]  # type: ignore[index]
        self.valid_masks = dataset.y_valid[:example_count]  # type: ignore[index]
        self.cell_size = cell_size
        self.image_size = dataset.x_train[0].shape[0]  # type: ignore[index]
        self.grid_size = self.image_size // self.cell_size

    def on_train_begin(self, epochs, logs=None):  # pylint: disable=W0613
        """Logs the ground truth at train_begin."""
        ground_truth = []
        for i, mask in enumerate(self.train_masks):
            plt.figure()
            plt.imshow(self.train_images[i])
            coord_list = get_coordinate_list(matrix=mask, size_image=self.image_size, size_grid=self.grid_size)
            plt.scatter(coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10)
            ground_truth.append(wandb.Image(plt, caption=f"Ground truth train: {i}"))
        wandb.log({f"Train ground truth": ground_truth}, commit=False)

        ground_truth_valid = []
        for i, mask in enumerate(self.valid_masks):
            plt.figure()
            plt.imshow(self.valid_images[i])
            coord_list = get_coordinate_list(matrix=mask, size_image=self.image_size, size_grid=self.grid_size)
            plt.scatter(coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10)
            ground_truth_valid.append(wandb.Image(plt, caption=f"Ground truth valid: {i}"))
        wandb.log({f"Valid ground truth": ground_truth_valid}, commit=False)

        plt.close(fig="all")

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=W0613
        """Logs predictions on epoch_end."""
        predictions_valid = []
        for i, image in enumerate(self.valid_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coord_list = get_coordinate_list(matrix=pred_mask, size_image=self.image_size, size_grid=self.grid_size)
            plt.scatter(coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10)
            predictions_valid.append(wandb.Image(plt, caption=f"Prediction: {i}"))
        wandb.log({f"Predictions valid dataset": predictions_valid}, commit=False)

        predictions_train = []
        for i, image in enumerate(self.train_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coord_list = get_coordinate_list(matrix=pred_mask, size_image=self.image_size, size_grid=self.grid_size)
            plt.scatter(coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10)
            predictions_train.append(wandb.Image(plt, caption=f"Prediction: {i}"))
        wandb.log({f"Predictions train dataset": predictions_train}, commit=False)

        plt.close(fig="all")


class DataShuffler(tf.keras.callbacks.Callback):
    """Temporary on_epoch_end data shuffling as tf v2.1.0 has the known bug of not calling on_epoch_end."""
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Reshuffle dataset."""
        # print("GLOBAL")
        # print([v for v in globals().keys() if not v.startswith('_')])
        # Access to the global variable "Dataset" here,
        # one should shuffle here or what I ended up doing
        # in the model base class added "shuffle=True" for ".fit"
        # this will only shuffle training data.


def train_model(model: Model, dataset: Dataset, cfg: Dict) -> Model:
    """Model training with wandb callbacks."""
    dataset_args = cfg["dataset_args"]
    wandb_callback = wandb.keras.WandbCallback()
    image_callback = WandbImageLogger(model, dataset, dataset_args["cell_size"])
    saver_callback = tf.keras.callbacks.ModelCheckpoint(
        f"../models/model_{cfg['name']}_{int(time.time())}.h5", save_best_only=False,
    )
    shuffle_callback = DataShuffler()
    callbacks = [wandb_callback, image_callback, saver_callback, shuffle_callback]

    tic = time.time()
    model.fit(dataset=dataset, callbacks=callbacks)
    print("Training took {:2f} s".format(time.time() - tic))

    return model


def run_experiment(cfg: Dict, save_weights: bool = False):
    """Run a training experiment.

    Args:
        cfg: Parsed yaml configuration file.
            Check the experiments folder for examples.
        save_weights: If model weights should be saved.
    """
    dataset_class_ = get_from_module("spot_detection.datasets", cfg["dataset"])
    model_class_ = get_from_module("spot_detection.models", cfg["model"])
    network_fn_ = get_from_module("spot_detection.networks", cfg["network"])
    optimizer_fn_ = get_from_module("spot_detection.optimizers", cfg["optimizer"])
    loss_fn_ = get_from_module("spot_detection.losses", cfg["loss"])

    network_args = cfg.get("network_args", {})
    dataset_args = cfg.get("dataset_args", {})
    train_args = cfg.get("train_args", {})

    dataset = dataset_class_(dataset_args["version"])
    dataset.load_data()

    model = model_class_(
        dataset_args=dataset_args,
        dataset_cls=dataset,
        loss_fn=loss_fn_,
        network_args=network_args,
        network_fn=network_fn_,
        optimizer_fn=optimizer_fn_,
        train_args=train_args,
    )

    cfg["system"] = {
        "gpus": tf.config.list_logical_devices("GPU"),
        "version": platform.version(),
        "platform": platform.platform(),
    }

    wandb.init(project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg)
    score = model.evaluate(dataset.x_valid, dataset.y_valid)
    wandb.log({"valid_metric": score})

    if save_weights:
        model.save_weights()

    wandb.join()
