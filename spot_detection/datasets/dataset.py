import pathlib

class Dataset:
    """ Simple abstract class for datasets. """

    @classmethod
    def data_dirname(cls):
        return pathlib.Path(__file__).resolve().parents[2] / "data"

    def load_or_generate_data(self):
        pass
