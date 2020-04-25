"""
Computes the baseline value metrics of spot detection given a dataset.
Returns a csv file with all metrics.
Here different metrics are used: f1_score, error on xy, and combination of f1_score and error on xy
"""

import argparse
import numpy as np
import os.path
import pandas as pd
import scipy.ndimage as ndi
import skimage.io
import skimage.exposure
import skimage.feature
import sys
import itertools
from typing import Tuple

sys.path.append("../")

import training.util_metrics
import training.util_trackmate
from training.util_prepare import get_prediction_matrix


def detect_spots(input_image: np.ndarray, cell_size: int) -> np.ndarray:
    """
    Use skimage.feature.blob_log to detect spots given an image.
    Return np.ndarray of shape (n, n, 3):
            p, x, y format for each cell

    Args:
        - input_image (np.ndarray): image used to detect spot
        - cell_size (int): size of cell used to calculate
            F1 score, precision and recall  
    """

    if not isinstance(input_image, np.ndarray):
        raise TypeError(f"input_image must be np.ndarray but is {type(input_image)}.")

    img = input_image
    img = skimage.exposure.equalize_hist(img, nbins=512)
    blobs = skimage.feature.blob_log(
        img, min_sigma=0.5, max_sigma=10, threshold=0.2, exclude_border=True
    )

    xy = np.stack([blobs[..., 1], blobs[..., 0]]).T
    xy = get_prediction_matrix(xy, len(img), cell_size)
    return xy


def compute_score(
    true: np.ndarray, pred: np.ndarray, cell_size: int, weight: float
) -> pd.DataFrame:
    """
    Compute F1 score, error on coordinate and a weighted average of the two.
    Return pd.DataFrame with scores

    Note – direction dependent, arguments cant be switched!!

    Args:
        - pred: list of np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: list of np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
        - cell_size: size of cells in the grid used to calculate
            F1 score, relative coordinates
        -weight: weight to on f1 score
    """
    f1_score = pd.Series(
        [training.util_metrics._f1_score(p, t) for p, t in zip(true, pred)]
    )

    err_coordinate = pd.Series(
        [
            training.util_metrics._error_on_coordinates(p, t, cell_size)
            for p, t in zip(true, pred)
        ]
    )

    weighted_f1_score_error_coordinates = pd.Series(
        [
            training.util_metrics._weighted_average_f1_score_error_coordinates(
                p, t, cell_size, weight
            )
            for p, t in zip(true, pred)
        ]
    )
    df = pd.DataFrame([f1_score, err_coordinate, weighted_f1_score_error_coordinates]).T
    df.columns = ["f1_score", "err_coordinate", "weighted_average"]
    return df


def _parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Path of the dataset folder."
    )
    parser.add_argument(
        "-z",
        "--cell_size",
        type=int,
        required=True,
        help="Size of the cell in the grid",
    )

    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        help="Value multiplied to f1_score to calculate single weighted score",
    )
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    cell_size = args.cell_size
    weight = 1
    if args.weight is not None:
        weight = args.weight

    print(f"Using dataset at {args.dataset}")

    with np.load(args.dataset, allow_pickle=True) as data:
        train_x = data["x_train"]
        valid_x = data["x_valid"]
        test_x = data["x_test"]
        train_y = data["y_train"]
        valid_y = data["y_valid"]
        test_y = data["y_test"]

    train_pred = list(map(detect_spots, train_x, itertools.repeat(cell_size)))
    valid_pred = list(map(detect_spots, valid_x, itertools.repeat(cell_size)))
    test_pred = list(map(detect_spots, test_x, itertools.repeat(cell_size)))

    for true, pred, name in zip(
        [train_y, valid_y, test_y],
        [train_pred, valid_pred, test_pred],
        ["train", "valid", "test"],
    ):
        df = compute_score(true=true, pred=pred, cell_size=cell_size, weight=weight)
        df_describe = df.describe()
        df_describe.to_csv(f"{os.path.splitext(args.dataset)[0]}.{name}_baseline.csv")


if __name__ == "__main__":
    main()
