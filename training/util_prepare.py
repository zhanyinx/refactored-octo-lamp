import importlib
import numpy as np
import os
from typing import Iterable, List


def extract_basename(path: dir) -> str:
    """ Returns the basename removing path and extension. """
    return os.path.splitext(os.path.basename(path))[0]


def train_valid_split(x_list: List[dir],
                      y_list: List[dir],
                      valid_split: float = 0.2,
                      shuffle: bool = True) -> Iterable[List[dir]]:
    """
    Splits two lists (input and predictions) into random training and validation
    sets with an optional shuffling.

    Args:
        - x_list: List containing filenames of all input.
        - y_list: List containing filenames of all predictions.
        - valid_split: Number between 0-1 to denote
            the percentage of examples used for validation.
    Returns:
        - x_train, x_valid, y_train, y_valid: Splited lists
            containing training or validation examples respectively.
    """
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(
            f"x_list, y_list must be list but is {type(x_list)}, {type(y_list)}.")
    if not isinstance(valid_split, float):
        raise TypeError(
            f"valid_split must be float but is {type(valid_split)}.")
    if not 0 <= valid_split <= 1:
        raise ValueError(
            f"valid_split must be between 0-1 but is {valid_split}.")

    if len(x_list) != len(y_list):
        raise ValueError(
            f"Lists must be of equal length: {len(x_list)} != {len(y_list)}.")
    if len(x_list) <= 2:
        raise ValueError("Lists must contain 2 elements or more.")

    if not all(os.path.exists(i) for i in x_list):
        raise OSError(f"x_list paths must exist.")
    if not all(os.path.exists(i) for i in y_list):
        raise OSError(f"y_list paths must exist.")

    def __shuffle(x_list: list, y_list: list):
        """ Shuffles two list keeping their relative arrangement. """
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
