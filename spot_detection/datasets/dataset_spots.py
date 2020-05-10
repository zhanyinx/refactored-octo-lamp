"""StandardDataset class."""

import os

import numpy as np

from .dataset import Dataset

DATA_DIRNAME = Dataset.data_dirname()


class SpotsDataset(Dataset):
    """Spots dataset class."""
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def data_filename(self) -> str:
        """Return the absolute path to dataset."""
        return f"{os.path.join(DATA_DIRNAME, self.name)}.npz"  # type: ignore[arg-type]

    def load_data(self) -> None:
        """Load dataset into memory."""
        with np.load(self.data_filename, allow_pickle=True) as data:
            self.x_train = data["x_train"]
            self.y_train = data["y_train"]
            self.x_valid = data["x_valid"]
            self.y_valid = data["y_valid"]
            self.x_test = data["x_test"]
            self.y_test = data["y_test"]
