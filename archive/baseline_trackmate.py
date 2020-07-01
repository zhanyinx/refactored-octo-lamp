"""Computes the baseline value metrics of spot detection given trackmate labels.

Returns a csv file with all metrics.
Here different metrics are used: f1_score, error on xy, and combination of f1_score and error on xy
"""


from typing import Iterable, List, Iterable
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import skimage.io

sys.path.append("../")
from spot_detection.io import extract_basename
from spot_detection.metrics import compute_score


def trackmate_get_file_lists(path: str) -> Iterable[List[str]]:
    """Extracts file paths and checks if respective files are present.

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
    """Create mask image with spot."""
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
    """Reads files groups, sorts them, convert coordinates to pixel unit and returns numpy arrays."""
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
    """Converts file lists into numpy arrays."""
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


def load_trackmate_data(path: str, size: int, cell_size: int, conversion: float = 1) -> Iterable[np.ndarray]:
    """Returns three lists of np.ndarray of shape (n,n,3): containing p,x,y.

    Args:
        - path (str): path to directory containing trackmate and labels subdirectories.
        - size (int): size of the images labeled by trackmate or human.
        - cell_size (int): size of cell used to calculate F1 score, precision and recall.
        - conversion (float): scaling factor used to convert coordinates into pixel unit (default = 1, no conversion).

    Returns:
        - images (np.ndarray): a numpy array of images.
        - label_true (np.ndarray): true mask (label or prediction)
        - label_trackmate (np.ndarray): trackmate prediction
    """
    if not all(isinstance(i, int) for i in (size, cell_size)):
        raise TypeError(f"size and cell_size must be int, but are {type(size), type(cell_size)}.")

    x_list, y_list, t_list = trackmate_get_file_lists(path)
    (images, label_true, label_trackmate,) = trackmate_files_to_numpy(
        x_list, y_list, t_list, conversion, size, cell_size
    )
    return images, label_true, label_trackmate


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trackmate",
        type=str,
        required=True,
        help="Path of directory containing labels and trackmate subfolders",
    )
    parser.add_argument(
        "-c",
        "--conversion",
        type=float,
        help="Rescaling factor to convert coordinates into pixel unit, required if --trackmate is defined",
    )
    parser.add_argument(
        "-s", "--size", type=int, required=True, help="Size of images",
    )
    parser.add_argument(
        "-z", "--cell_size", type=int, required=True, help="Size of the cell in the grid",
    )

    parser.add_argument(
        "-w", "--weight", type=float, help="Value multiplied to f1_score to calculate single weighted score",
    )
    args = parser.parse_args()

    return args


def main():
    """Computes baseline for spots from trackmate."""
    args = _parse_args()

    size = args.size
    cell_size = args.cell_size
    weight = 1

    if args.weight is not None:
        weight = args.weight

    print(f"Using trackmate data at {args.trackmate}")

    if args.conversion is None:
        raise ValueError("--trackmate requires --conversion.")

    trackmate = args.trackmate
    conversion = args.conversion

    images, label_true, label_trackmate = load_trackmate_data(
        path=trackmate, conversion=conversion, size=size, cell_size=cell_size
    )
    df = compute_score(true=label_true, pred=label_trackmate, cell_size=cell_size, weight=weight)
    df_describe = df.describe()
    df_describe.to_csv(f"{os.path.splitext(args.trackmate)[0]}_trackmate_baseline.csv")


if __name__ == "__main__":
    main()
