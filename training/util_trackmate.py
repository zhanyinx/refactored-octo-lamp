""" Utils necessary to load data from labels/ trackmate/ folder structure."""

import glob
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("../")
from typing import List, Tuple

from training.util_prepare import extract_basename


def trackmate_get_file_lists(path: dir) -> Tuple[List[dir]]:
    """
    Extracts file paths and checks if respective files are present.

    Args:
        - Path: Relative or absolute location of directory containing
            images and labels subdirectories.

    Returns:
        - y_list, t_list: Lists of absolute file paths for the files found
            in the labels and trackmate subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")
    if not all(os.path.exists(os.path.join(path, i)) for i in ["labels", "trackmate"]):
        raise OSError(
            f"Path {path} must contain an labels/ and trackmate subdirectory."
        )

    y_list = sorted(glob.glob(f"{os.path.join(path, 'labels')}/*.txt"))
    t_list = sorted(glob.glob(f"{os.path.join(path, 'trackmate')}/*.csv"))

    if not len(t_list) == len(y_list):
        raise ValueError(
            f"Length of trackmate/ and labels/ must match: {len(t_list)} != {len(y_list)}."
        )
    if len(y_list) == 0:
        raise ValueError(f"No files found in path {path}.")

    y_basenames = [extract_basename(f) for f in y_list]
    t_basenames = [extract_basename(f) for f in t_list]

    if not all((x == y) for x, y in zip(t_basenames, y_basenames)):
        raise ValueError(f"Names of trackmate, and labels/ files must match.")

    y_list = [os.path.abspath(f) for f in y_list]
    t_list = [os.path.abspath(f) for f in t_list]

    return y_list, t_list


def trackmate_create_spot_mask(
    spot_coord: np.ndarray, size: int, cell_size: int
) -> np.ndarray:
    """Create mask image with spot"""
    img = np.zeros((size // cell_size, size // cell_size, 3))
    for i in range(len(spot_coord)):
        x = int(np.floor(spot_coord[i, 0])) // cell_size
        y = int(np.floor(spot_coord[i, 1])) // cell_size
        img[x, y, 0] = 1
        img[x, y, 1] = spot_coord[i, 0]
        img[x, y, 2] = spot_coord[i, 1]

    return img


def trackmate_group_to_numpy(
    label: dir, trackmate: dir, conversion: float, size: int, cell_size: int
) -> Tuple[np.ndarray]:
    """ Reads files groups, sorts them, convert coordinates to pixel unit and returns numpy arrays. 
    """

    df = pd.read_table(label)
    df_trackmate = pd.read_csv(trackmate)

    if len(df) < 5:
        return 0, 0, 0

    df.columns = ["x", "y"]
    df_trackmate.columns = ["number", "x", "y"]

    xy = np.stack([df["x"].to_numpy(), df["y"].to_numpy()]).T
    xy_trackmate = np.stack(
        [df_trackmate["x"].to_numpy(), df_trackmate["y"].to_numpy()]
    ).T

    xy = xy * conversion
    xy_trackmate = xy_trackmate * conversion

    xy = trackmate_create_spot_mask(xy, size, cell_size)
    xy_trackmate = trackmate_create_spot_mask(xy_trackmate, size, cell_size)

    return xy, xy_trackmate


def trackmate_remove_zeros(lst: list) -> list:
    """ Removes all occurences of "0" from a list. """
    return [i for i in lst if i is not 0]


def trackmate_files_to_numpy(
    labels: List[dir],
    trackmates: List[dir],
    conversion: float,
    size: int,
    cell_size: int,
) -> Tuple[np.ndarray]:
    """ Converts file lists into numpy arrays. """
    np_labels = []
    np_trackmate = []

    for label, trackmate in zip(labels, trackmates):
        label, trackmate = trackmate_group_to_numpy(
            label, trackmate, conversion, size, cell_size
        )
        np_labels.append(label)
        np_trackmate.append(trackmate)

    np_labels = trackmate_remove_zeros(np_labels)
    np_trackmate = trackmate_remove_zeros(np_trackmate)

    return np_labels, np_trackmate
