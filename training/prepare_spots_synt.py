"""
Prepares a dataset from an folder containing .tif images and .csv labels structure.
As usually generated synthetically in Fiji.
"""

from typing import List, Tuple
import argparse
import glob
import numpy as np
import os
import pandas as pd
import secrets
import skimage.io
import sys


sys.path.append("../")
from training.util_prepare import (
    extract_basename,
    train_valid_split,
    get_prediction_matrix,
)


def get_file_lists(path: dir) -> Tuple[List[dir]]:
    """
    Extracts file paths and checks if respective files are present.

    Args:
        - Path: Relative or absolute location of directory containing
            images and masks subdirectories.

    Returns:
        - x_list, y_list: Lists of absolute file paths for the files found
            in the images or masks subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")

    x_list = sorted(glob.glob(f"{path}*.tif"))
    y_list = sorted(glob.glob(f"{path}*.csv"))

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


def import_data(
    x_list: List[dir], y_list: List[dir]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """ Opens files from lists as images and DataFrames. """

    images, labels = [], []
    n_images = 0
    for x, y in zip(x_list, y_list):

        # Image import
        image = skimage.io.imread(x)
        assert image.ndim == 3
        images.append(image)

        # Label import
        df = pd.read_csv(y)
        for _, row in df.iterrows():
            labels.append(
                [row["t [sec]"] + n_images, row["y [pixel]"], row["x [pixel]"]]
            )
        n_images += min(image.shape)

    images = np.concatenate(images)
    df = pd.DataFrame(labels, columns=["img_index", "x", "y"])

    return images, df


def files_to_numpy(
    images: List[dir], labels: List[dir], cell_size: int = 4
) -> Tuple[np.ndarray]:
    """ Converts file lists into numpy arrays. """
    np_images, labels = import_data(images, labels)

    np_labels = []
    for i in labels["img_index"].unique():
        curr_df = labels[labels["img_index"] == i].reset_index(drop=True)
        xy = np.stack([curr_df["x"].to_numpy(), curr_df["y"].to_numpy()]).T
        np_labels.append(get_prediction_matrix(xy, 512, cell_size))

    np_images /= 255
    np_labels = np.array(np_labels)

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
        required=False,
        help="Basename of dataset.",
    )
    parser.add_argument(
        "-s",
        "--cell_size",
        type=int,
        default=4,
        required=False,
        help="Size of the prediction cells.",
    )
    parser.add_argument(
        "-t",
        "--test_split",
        type=int,
        default=0.1,
        required=False,
        help="Percentage split of test set.",
    )
    parser.add_argument(
        "-v",
        "--valid_split",
        type=int,
        default=0.2,
        required=False,
        help="Precentage split of validation set.",
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

    x_train, y_train = files_to_numpy(x_train, y_train, args.cell_size)
    x_valid, y_valid = files_to_numpy(x_valid, y_valid, args.cell_size)
    x_test, y_test = files_to_numpy(x_test, y_test, args.cell_size)

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
