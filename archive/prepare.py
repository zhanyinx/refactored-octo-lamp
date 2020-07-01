"""Dataset preparation functions."""

import argparse
import glob
import os
import sys
from typing import Tuple

sys.path.append("../")
from spot_detection.io import extract_basename


def get_file_lists(path: str, format_image: str, format_label: str) -> Tuple[List[str], List[str]]:
    """Extracts file paths and checks if respective files are present.

    Args:
        Path: Relative or absolute location of directory containing
            images and masks subdirectories.
        format_image: Format of the image (e.g. tif, png, etc...)
        format_label: Format of the label (e.g. csv, txt, etc...)

    Returns:
        x_list, y_list: Lists of absolute file paths for the files found
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


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path of the dataset folder.")
    parser.add_argument(
        "-b", "--basename", type=str, default="ds", required=True, help="Basename of dataset.",
    )
    parser.add_argument(
        "-z", "--cell_size", type=int, default=4, required=True, help="Size of cell in the grid for making y_true",
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
