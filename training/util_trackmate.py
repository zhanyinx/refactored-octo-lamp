"""Utils necessary to load data from labels/ trackmate/ folder structure."""

import glob
import math
import os
import sys
from typing import List, Iterable

import numpy as np
import pandas as pd
import skimage.io

sys.path.append("../")
from training.util_prepare import extract_basename


def trackmate_get_file_lists(path: str) -> Iterable[List[str]]:
    """
    Extracts file paths and checks if respective files are present.

    Args:
        - Path: Relative or absolute location of directory containing
            images and labels subdirectories.

    Returns:
        - x_list, y_list, t_list: Lists of absolute file paths for the files found
            in the images, labels and trackmate subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")
    if not all(os.path.exists(os.path.join(path, i)) for i in ["images", "labels", "trackmate"]):
        raise OSError(f"Path {path} must contain an images/ , labels/ and trackmate subdirectory.")

    x_list = sorted(glob.glob(f"{os.path.join(path, 'images')}/*.tif"))
    y_list = sorted(glob.glob(f"{os.path.join(path, 'labels')}/*.txt"))
    t_list = sorted(glob.glob(f"{os.path.join(path, 'trackmate')}/*.csv"))

    if not len(t_list) == len(y_list):
        raise ValueError(f"Length of trackmate/ and labels/ must match: {len(t_list)} != {len(y_list)}.")

    if not len(x_list) == len(y_list):
        raise ValueError(f"Length of images/ and labels/ must match: {len(x_list)} != {len(y_list)}.")

    if len(y_list) == 0:
        raise ValueError(f"No files found in path {path}.")

    x_basenames = [extract_basename(f) for f in x_list]
    y_basenames = [extract_basename(f) for f in y_list]
    t_basenames = [extract_basename(f) for f in t_list]

    if not all((t == y) for t, y in zip(t_basenames, y_basenames)):
        raise ValueError(f"Names of trackmate, and labels/ files must match.")

    if not all((x == y) for x, y in zip(x_basenames, y_basenames)):
        raise ValueError(f"Names of images, and labels files must match.")

    x_list = [os.path.abspath(f) for f in x_list]
    y_list = [os.path.abspath(f) for f in y_list]
    t_list = [os.path.abspath(f) for f in t_list]

    return x_list, y_list, t_list


def trackmate_create_spot_mask(spot_coord: np.ndarray, size: int, cell_size: int) -> np.ndarray:
    """Create mask image with spot"""
    pred = np.zeros((math.ceil(size / cell_size), math.ceil(size / cell_size), 3))
    for nspot in range(len(spot_coord)):
        i = int(np.floor(spot_coord[nspot, 0])) // cell_size
        j = int(np.floor(spot_coord[nspot, 1])) // cell_size
        rel_x = (spot_coord[nspot, 0] - i * cell_size) / cell_size
        rel_y = (spot_coord[nspot, 1] - j * cell_size) / cell_size
        pred[i, j, 0] = 1
        pred[i, j, 1] = rel_x
        pred[i, j, 2] = rel_y

    return pred


def trackmate_group_to_numpy(
    image: str, label: str, trackmate: str, conversion: float, size: int, cell_size: int
) -> Iterable[np.ndarray]:
    """ Reads files groups, sorts them, convert coordinates to pixel unit and returns numpy arrays."""

    image = skimage.io.imread(image)
    df = pd.read_table(label)
    df_trackmate = pd.read_csv(trackmate)

    if len(df) < 5:
        return 0, 0, 0  # type: ignore[return-value]

    df.columns = ["y", "x"]
    df_trackmate.columns = ["number", "y", "x"]

    xy = np.stack([df["x"].to_numpy(), df["y"].to_numpy()]).T
    xy_trackmate = np.stack([df_trackmate["x"].to_numpy(), df_trackmate["y"].to_numpy()]).T

    xy = xy * conversion
    xy_trackmate = xy_trackmate * conversion

    xy = trackmate_create_spot_mask(xy, size, cell_size)
    xy_trackmate = trackmate_create_spot_mask(xy_trackmate, size, cell_size)

    return image, xy, xy_trackmate


def trackmate_remove_zeros(lst: list) -> list:
    """Removes all occurences of "0" from a list."""
    return [i for i in lst if isinstance(i, np.ndarray)]


def trackmate_files_to_numpy(
    images: List[str], labels: List[str], trackmates: List[str], conversion: float, size: int, cell_size: int,
) -> Iterable[np.ndarray]:
    """ Converts file lists into numpy arrays."""
    np_images = []
    np_labels = []
    np_trackmate = []

    for image, label, trackmate in zip(images, labels, trackmates):
        image, label, trackmate = trackmate_group_to_numpy(image, label, trackmate, conversion, size, cell_size)
        np_images.append(image)
        np_labels.append(label)
        np_trackmate.append(trackmate)

    np_images = trackmate_remove_zeros(np_images)
    np_labels = trackmate_remove_zeros(np_labels)
    np_trackmate = trackmate_remove_zeros(np_trackmate)

    return np_images, np_labels, np_trackmate
