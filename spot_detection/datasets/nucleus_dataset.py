import numpy as np
import os

from .dataset import Dataset

DATA_DIRNAME = Dataset.data_dirname()


class SpotDataset(Dataset):
    """
    Each dataset type has its own class. Allows for dataset specific loading / preprocessing etc.
    NOT the actual class passed into for fitting – DatasetSequence.
    """

    def __init__(self, name: str):
        self.name = name
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    def __name__(self):
        return "nuclei"

    @property
    def data_filename(self):
        return f"{os.path.join(DATA_DIRNAME, self.name)}.npz"
    
    def load_data(self):
        with np.load(self.data_filename) as data:
            self.x_train = data["x_train"]
            self.x_valid = data["x_valid"]
            self.y_train = data["y_train"]
            self.y_valid = data["y_valid"]
