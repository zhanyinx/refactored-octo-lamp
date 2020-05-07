"""Model class, to be extended by specific types of models."""

import pathlib
import datetime
from typing import Callable, Dict

import numpy as np

from spot_detection.datasets.dataset import Dataset
from spot_detection.datasets.dataset_sequence import DatasetSequence

DIRNAME = pathlib.Path(__file__).parents[1].resolve() / "weights"
DATESTRING = datetime.datetime.now().strftime("%Y%d%m_%H%M")


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(
        self,
        dataset_args: Dict,
        dataset_cls: Dataset,
        network_args: Dict,
        network_fn: Callable,
        loss_fn: Callable,
        optimizer_fn: Callable,
        train_args: Dict,
        batch_format_fn: Callable = None,
        batch_augment_fn: Callable = None,
    ):
        self.name = f"{DATESTRING}_{self.__class__.__name__}_{dataset_cls.name}_{network_fn.__name__}"

        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        self.network = network_fn(n_channels=network_args["n_channels"])  # **network_args
        self.dataset_args = dataset_args
        self.train_args = train_args
        self.batch_format_fn = batch_format_fn
        self.batch_augment_fn = batch_augment_fn

        try:
            self.load_weights()
        except KeyError:
            print("Training from scratch.")

    @property
    def weights_filename(self) -> str:
        """Return the absolute path to weight file."""
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    @property
    def metrics(self) -> list:
        """Return metrics."""
        return ["accuracy"]

    def fit(self, dataset: Dataset, augment_val: bool = True, callbacks: list = None,) -> None:
        """Training loop."""
        if callbacks is None:
            callbacks = []

        self.network.compile(
            loss=self.loss_fn,
            optimizer=self.optimizer_fn(float(self.train_args["learning_rate"])),
            metrics=self.metrics,
        )

        train_sequence = DatasetSequence(
            dataset.x_train,
            dataset.y_train,
            self.train_args["batch_size"],
            format_fn=self.batch_format_fn,
            augment_fn=self.batch_augment_fn,
        )
        valid_sequence = DatasetSequence(
            dataset.x_valid,
            dataset.y_valid,
            self.train_args["batch_size"],
            format_fn=self.batch_format_fn,
            augment_fn=self.batch_augment_fn if augment_val else None,
        )

        self.network.fit(
            train_sequence,
            epochs=self.train_args["epochs"],
            callbacks=callbacks,
            validation_data=valid_sequence,
            shuffle=True,
            # use_multiprocessing=False,
            # workers=1,
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model."""
        sequence = DatasetSequence(x, y, batch_size=self.train_args["batch_size"])
        preds = self.network.predict(sequence)
        return np.mean(np.square(preds) - np.square(y))

    def load_weights(self) -> None:
        """Load model weights."""
        self.network.load_weights(self.train_args["pretrained"])

    def save_weights(self) -> None:
        """Save model weights."""
        self.network.save_weights(self.weights_filename)
