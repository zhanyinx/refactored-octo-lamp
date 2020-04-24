import numpy as np
import os

from .dataset import Dataset

DATA_DIRNAME = Dataset.data_dirname()


class SpotsDataset(Dataset):
    """
    Spots dataset class.
    """

    def __init__(self, name: str):
        self.name = name
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    def __name__(self) -> str:
        return "spots"

    @property
    def data_filename(self) -> dir:
        return f"{os.path.join(DATA_DIRNAME, self.name)}.npz"
    
    def load_data(self) -> None:
        with np.load(self.data_filename, allow_pickle=True) as data:
            self.x_train = data["x_train"]
            self.x_valid = data["x_valid"]
            self.y_train = data["y_train"]
            self.y_valid = data["y_valid"]
