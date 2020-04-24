""" Prepares a dataset from an images/ labels/ folder structure. """

import argparse
import glob
import numpy as np
import os
import pandas as pd
import secrets
import skimage.io
import sys

sys.path.append("../")
from typing import List, Tuple

from training.util_prepare import extract_basename, train_valid_split, _create_spot_mask


def get_file_lists(path: dir) -> Tuple[List[dir]]:
    """
    Extracts file paths and checks if respective files are present.

    Args:
        - Path: Relative or absolute location of directory containing
            images and labels subdirectories.

    Returns:
        - x_list, y_list: Lists of absolute file paths for the files found
            in the images or labels subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")
    if not all(os.path.exists(os.path.join(path, i)) for i in ["images", "labels"]):
        raise OSError(f"Path {path} must contain an images/ and labels/ subdirectory.")

    x_list = sorted(glob.glob(f"{os.path.join(path, 'images')}/*.tif"))
    y_list = sorted(glob.glob(f"{os.path.join(path, 'labels')}/*.txt"))

    if not len(x_list) == len(y_list):
        raise ValueError(
            f"Length of images/ and labels/ must match: {len(x_list)} != {len(y_list)}."
        )
    if len(x_list) == 0:
        raise ValueError(f"No files found in path {path}.")

    x_basenames = [extract_basename(f) for f in x_list]
    y_basenames = [extract_basename(f) for f in y_list]

    if not all((x == y) for x, y in zip(x_basenames, y_basenames)):
        raise ValueError(f"Names of images/ and labels/ files must match.")

    x_list = [os.path.abspath(f) for f in x_list]
    y_list = [os.path.abspath(f) for f in y_list]

    return x_list, y_list


def group_to_numpy(
    image: dir, label: dir, conversion: float, cell_size: int, bitdepth: int
) -> Tuple[np.ndarray]:
    """ Reads files groups, sorts them, convert coordinates to pixel unit and returns numpy arrays. 
    """

    image = skimage.io.imread(image)
    image /= 2 ** bitdepth - 1  # normalise

    df = pd.read_table(label)
    if min(image.shape) < 512 or len(df) < 5:
        return 0, 0

    df.columns = ["x", "y"]
    xy = np.stack([df["x"].to_numpy(), df["y"].to_numpy()]).T
    xy = xy * conversion
    xy = _create_spot_mask(xy, len(image), cell_size)

    return image, xy


def remove_zeros(lst: list) -> list:
    """ Removes all occurences of "0" from a list. """
    return [i for i in lst if i is not 0]


def files_to_numpy(
    images: List[dir],
    labels: List[dir],
    conversion: float,
    cell_size: int,
    bitdepth: int,
) -> Tuple[np.ndarray]:
    """ Converts file lists into numpy arrays. """
    np_images = []
    np_labels = []

    for image, label in zip(images, labels):
        image, label = group_to_numpy(image, label, conversion, cell_size, bitdepth)
        np_images.append(image)
        np_labels.append(label)

    np_images = remove_zeros(np_images)
    np_labels = remove_zeros(np_labels)

    return np_images, np_labels


def _parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path of the dataset folder.")
    parser.add_argument(
        "-b",
        "--basename",
        type=str,
        default="ds",
        required=True,
        help="Basename of dataset.",
    )
    parser.add_argument(
        "-z",
        "--cell_size",
        type=int,
        default=4,
        required=True,
        help="Size of cell in the grid for making y_true",
    )
    parser.add_argument(
        "-d",
        "--bitdepth",
        type=int,
        default=1,
        required=True,
        help="Bitdepth of image",
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
        "-t",
        "--test_split",
        type=float,
        default=0.1,
        required=False,
        help="Fraction split of test set.",
    )
    parser.add_argument(
        "-v",
        "--valid_split",
        type=float,
        default=0.2,
        required=False,
        help="Fraction split of validation set.",
    )
    args = parser.parse_args()

    return args


def main():
    """ Parse command-line argument and prepare dataset. """
    args = _parse_args()

    x_list, y_list = get_file_lists(args.path)
    x_trainval, x_test, y_trainval, y_test = train_valid_split(
        x_list=x_list, y_list=y_list, valid_split=args.test_split
    )
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x_list=x_trainval, y_list=y_trainval, valid_split=args.valid_split
    )

    x_train, y_train = files_to_numpy(
        x_train, y_train, args.conversion, args.cell_size, args.bitdepth
    )
    x_valid, y_valid = files_to_numpy(
        x_valid, y_valid, args.conversion, args.cell_size, args.bitdepth
    )
    x_test, y_test = files_to_numpy(
        x_test, y_test, args.conversion, args.cell_size, args.bitdepth
    )

    print(f"All files*: {len(x_list)}")
    print(f"All files: {len(x_train) + len(x_valid) + len(x_test)}")
    print(f"  - Train: {len(x_train)}")
    print(f"  - Valid: {len(x_valid)}")
    print(f"  - Test: {len(x_test)}")

    fname = f"../data/{args.basename}_{secrets.token_hex(4)}.npz"
    np.savez_compressed(
        fname,
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
