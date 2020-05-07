import importlib
import numpy as np
import os
from typing import Iterable, List
import argparse
from typing import Tuple
import glob
import math


def get_file_lists(path: str, format_image: str, format_label: str) -> Tuple[List[str], List[str]]:
    """
    Extracts file paths and checks if respective files are present.
    Args:
        - Path: Relative or absolute location of directory containing
            images and masks subdirectories.
        - format_image: Format of the image (e.g. tif, png, etc...)
        - format_label: Format of the label (e.g. csv, txt, etc...)
    Returns:
        - x_list, y_list: Lists of absolute file paths for the files found
            in the images or masks subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")
    if not all(os.path.exists(os.path.join(path, i)) for i in ["images", "labels"]):
        raise OSError(f"Path {path} must contain an images/ and labels/ subdirectory.")

    x_list = sorted(glob.glob(f"{os.path.join(path, 'images')}/*.{format_image}"))
    y_list = sorted(glob.glob(f"{os.path.join(path, 'labels')}/*.{format_label}"))

    if not len(x_list) == len(y_list):
        raise ValueError(f"Length of images/ and labels/ must match: {len(x_list)} != {len(y_list)}.")
    if len(x_list) == 0:
        raise ValueError(f"No files found in path {path}.")

    x_basenames = [extract_basename(f) for f in x_list]
    y_basenames = [extract_basename(f) for f in y_list]

    if not all((x == y) for x, y in zip(x_basenames, y_basenames)):
        raise ValueError(f"Names of images/ and labels/ files must match.")

    x_list = [os.path.abspath(f) for f in x_list]
    y_list = [os.path.abspath(f) for f in y_list]

    return x_list, y_list


def remove_zeros(lst: list) -> list:
    """ Removes all occurences of "0" from a list. """
    return [i for i in lst if i is not 0]


def _parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path of the dataset folder.")
    parser.add_argument(
        "-b", "--basename", type=str, default="ds", required=True, help="Basename of dataset.",
    )
    parser.add_argument(
        "-z", "--cell_size", type=int, default=4, required=True, help="Size of cell in the grid for making y_true",
    )
    parser.add_argument(
        "-d", "--bitdepth", type=int, default=1, required=False, help="Bitdepth of image",
    )
    parser.add_argument(
        "-i", "--image_format", type=str, default="tif", required=False, help="Format of images (e.g. tif, png)",
    )
    parser.add_argument(
        "-l", "--label_format", type=str, default="csv", required=False, help="Format of labels (e.g. csv, txt)",
    )
    parser.add_argument(
        "-c",
        "--conversion",
        type=float,
        default=1,
        required=False,
        help="Rescaling factor to convert coordinates into pixel unit",
    )
    parser.add_argument(
        "-t", "--test_split", type=float, default=0.1, required=False, help="Fraction split of test set.",
    )
    parser.add_argument(
        "-v", "--valid_split", type=float, default=0.2, required=False, help="Fraction split of validation set.",
    )
    args = parser.parse_args()

    return args


def extract_basename(path: str) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


def get_prediction_matrix(spot_coord: np.ndarray, size: int, cell_size: int, size_y: int = None) -> np.ndarray:
    """Return np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Args:
        spot_coord: List of coordinates in x, y format with shape (n, 2).
        size: size of the image from which List of coordinates are extracted.
        cell_size: size of cell used to calculate F1 score, precision and recall.
        size_y: if not provided, it assumes it is squared image, otherwise the second shape of image
    """

    if not all(isinstance(i, int) for i in (size, cell_size)):
        raise TypeError(f"size and cell_size must be int, but are {type(size), type(cell_size)}.")

    nrow = math.ceil(size / cell_size)
    ncol = nrow
    if size_y is not None:
        ncol = math.ceil(size_y / cell_size)

    pred = np.zeros((nrow, ncol, 3))
    for s in range(len(spot_coord)):
        i = int(np.floor(spot_coord[s, 0])) // cell_size
        j = int(np.floor(spot_coord[s, 1])) // cell_size
        rel_x = (spot_coord[s, 0] - i * cell_size) / cell_size
        rel_y = (spot_coord[s, 1] - j * cell_size) / cell_size
        pred[i, j, 0] = 1
        pred[i, j, 1] = rel_x
        pred[i, j, 2] = rel_y

    return pred


def train_valid_split(
    x_list: List[str], y_list: List[str], valid_split: float = 0.2, shuffle: bool = True
) -> Iterable[List[str]]:
    """Split two lists (input and predictions).
    Splitting into random training and validation sets with an optional shuffling.

    Args:
        x_list: List containing filenames of all input.
        y_list: List containing filenames of all predictions.
        valid_split: Number between 0-1 to denote the percentage of examples used for validation.
    Returns:
        x_train, x_valid, y_train, y_valid: Splited lists containing training or validation examples respectively.
    """
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f"x_list, y_list must be list but is {type(x_list)}, {type(y_list)}.")
    if not isinstance(valid_split, float):
        raise TypeError(f"valid_split must be float but is {type(valid_split)}.")
    if not 0 <= valid_split <= 1:
        raise ValueError(f"valid_split must be between 0-1 but is {valid_split}.")

    if len(x_list) != len(y_list):
        raise ValueError(f"Lists must be of equal length: {len(x_list)} != {len(y_list)}.")
    if len(x_list) <= 2:
        raise ValueError("Lists must contain 2 elements or more.")

    if not all(os.path.exists(i) for i in x_list):
        raise OSError(f"x_list paths must exist.")
    if not all(os.path.exists(i) for i in y_list):
        raise OSError(f"y_list paths must exist.")

    def __shuffle(x_list: list, y_list: list):
        """Shuffles two list keeping their relative arrangement."""
        import random

        combined = list(zip(x_list, y_list))
        random.shuffle(combined)
        x_tuple, y_tuple = zip(*combined)
        return list(x_tuple), list(y_tuple)

    if shuffle:
        x_list, y_list = __shuffle(x_list, y_list)

    split_len = round(len(x_list) * valid_split)

    x_valid = x_list[:split_len]
    x_train = x_list[split_len:]
    y_valid = y_list[:split_len]
    y_train = y_list[split_len:]

    return x_train, x_valid, y_train, y_valid
