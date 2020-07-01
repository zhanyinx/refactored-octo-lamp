"""SpotsDataset class."""

import os

from ._datasets import Dataset
from ..io import load_npz

DATA_DIRNAME = Dataset.data_dirname()


class SpotsDataset(Dataset):
    """Class used to load all spots data."""

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def data_filename(self) -> str:
        """Return the absolute path to dataset."""
        return f"{os.path.join(DATA_DIRNAME, self.name)}.npz"  # type: ignore[arg-type]

    def load_data(self) -> None:
        """Load dataset into memory."""
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = load_npz(self.data_filename)
