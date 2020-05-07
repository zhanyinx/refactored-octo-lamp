import pathlib


class Dataset:
    """ Simple abstract class for datasets. """

    def __init__(self, name: str = None):
        self.name = name
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    @classmethod
    def data_dirname(cls):
        return pathlib.Path(__file__).resolve().parents[2] / "data"

    def load_or_generate_data(self):
        pass
