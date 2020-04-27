import importlib
import platform
import sys
import tensorflow as tf
import time
import wandb
from typing import Dict
import matplotlib.pyplot as plt
sys.path.append("../")

from spot_detection.datasets.dataset import Dataset
from spot_detection.models.base import Model
from util_prediction import get_coordinate_list, get_relative_coordinates, get_absolute_coordinates


def get_from_module(path: str, attribute: str) -> type:
    """ Grabs an attribute from a given module path. """
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute


class WandbImageLogger(tf.keras.callbacks.Callback):
    """
    Custom image prediction logger callback in wandb.
    Expects segmentation images and the model class to have a predict_on_image method.
    """

    def __init__(self, model_wrapper: Model, dataset: Dataset, example_count: int = 4):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.valid_images = dataset.x_valid[:example_count]
        self.train_images = dataset.x_train[:example_count]
        self.valid_masks = dataset.y_train[:example_count]

    def on_epoch_end(self, epoch, logs=None):

        ground_truth = []
        for i, mask in enumerate(self.valid_masks):
            plt.figure()
            plt.imshow(self.valid_images[i])
            coordList = get_coordinate_list(matrix = mask, size_image = 512, size_grid = 128)
            plt.scatter(coordList[...,0], coordList[...,1], marker = "+", color = "r", s = 10)
            ground_truth.append(wandb.Image(plt, caption=f"Ground truth: {i}"))

        wandb.log({f"Ground truth": ground_truth}, commit=False)
        

        predictions_valid = []
        for i, image in enumerate(self.valid_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coordList = get_coordinate_list(matrix = pred_mask, size_image = 512, size_grid = 128)
            plt.scatter(coordList[...,0], coordList[...,1], marker = "+", color = "r", s = 10)
            predictions_valid.append(wandb.Image(plt, caption=f"Prediction: {i}"))

        wandb.log({f"Predictions valid dataset {i}": predictions_valid}, commit=False)


        predictions_train = []
        for i, image in enumerate(self.train_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coordList = get_coordinate_list(matrix = pred_mask, size_image = 512, size_grid = 128)
            plt.scatter(coordList[...,0], coordList[...,1], marker = "+", color = "r", s = 10)
            predictions_train.append(wandb.Image(plt, caption=f"Prediction: {i}"))

        wandb.log({f"Predictions train dataset {i}": predictions_train}, commit=False)


        plt.close()

      
        #ground_truth = [
        #    wandb.Image(image,
        #                caption=f"Ground truth: {i}")
        #    for i, image in enumerate(self.valid_masks)
        #]
        #wandb.log({"Ground truth": ground_truth}, commit=False)

        #predictions = [
        #    wandb.Image(self.model_wrapper.predict_on_image(image),
        #                caption=f"Prediction: {i}")
        #    for i, image in enumerate(self.valid_images)
        #]
        #wandb.log({"Predictions": predictions}, commit=False)


class DataShuffler(tf.keras.callbacks.Callback):
    """
    Temporary on_epoch_end data shuffling as tf v2.1.0 has the known bug
    of not calling on_epoch_end for tf.keras.utils.Sequence classes.
    """

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # print("GLOBAL")
        # print([v for v in globals().keys() if not v.startswith('_')])
        # Access to the global variable "Dataset" here,
        # one should shuffle here or what I ended up doing
        # in the model base class added "shuffle=True" for ".fit"
        # this will only shuffle training data.
        pass


def train_model(model: Model, dataset: Dataset) -> Model:
    """ Model training with wandb callbacks. """

    wandb_callback = wandb.keras.WandbCallback()
    image_callback = WandbImageLogger(model, dataset)
    saver_callback = tf.keras.callbacks.ModelCheckpoint(
        f"../models/model_{int(time.time())}.h5", save_best_only=False,
    )
    shuffle_callback = DataShuffler()
    callbacks = [wandb_callback, image_callback,
                 saver_callback, shuffle_callback]

    tic = time.time()
    _ = model.fit(dataset=dataset, callbacks=callbacks)
    print("Training took {:2f} s".format(time.time() - tic))

    return model


def run_experiment(cfg: Dict, save_weights: bool = False):
    """
    Runs a training experiment.
    Args:
        - cfg: Read yaml configuration file with format:
            dataset: "TestDataset"
            dataset_args:
                version: "dataset_hash"
            model: "ExampleModel"
            network: "fcn8"
            network_args:
                layers: 4
                width: 4
            loss: "binary_crossentropy"
            optimizer: "adam"
            train_args:
                batch_size: 4
                epochs: 10
                learning_rate: 3e-04
        - save_weights: If model weights should be saved.
    """

    dataset_class_ = get_from_module("spot_detection.datasets", cfg["dataset"])
    model_class_ = get_from_module("spot_detection.models", cfg["model"])
    network_fn_ = get_from_module("spot_detection.networks", cfg["network"])
    optimizer_fn_ = get_from_module(
        "spot_detection.optimizers", cfg["optimizer"])
    loss_fn_ = get_from_module("spot_detection.losses", cfg["loss"])

    network_args = cfg.get("network_args", {})
    dataset_args = cfg.get("dataset_args", {})
    train_args = cfg.get("train_args", {})

    dataset = dataset_class_(dataset_args["version"])
    dataset.load_data()

    model = model_class_(
        dataset_cls=dataset,
        network_fn=network_fn_,
        loss_fn=loss_fn_,
        optimizer_fn=optimizer_fn_,
        train_args=train_args,
        dataset_args=dataset_args,
        network_args=network_args,
    )

    cfg["system"] = {
        "gpus": tf.config.list_logical_devices("GPU"),
        "version": platform.version(),
        "platform": platform.platform(),
    }

    wandb.init(project=cfg["name"], config=cfg)

    model = train_model(model, dataset)
    score = model.evaluate(dataset.x_valid, dataset.y_valid)
    wandb.log({"valid_metric": score})

    if save_weights:
        model.save_weights()

    wandb.join()
