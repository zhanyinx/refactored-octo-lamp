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
import skimage.filters
import skimage.io
import skimage.exposure
import skimage.feature
import skimage.morphology
import sys
from typing import Tuple

sys.path.append("../")

import training.util_metrics
import training.util_trackmate


def load_trackmate_data(path: dir, conversion: float = 1) -> Tuple[np.ndarray]:
    """ Parse command-line argument and prepare dataset. """

    y_list, t_list = training.util_trackmate.get_file_lists(path)
    label_true, label_trackmate = training.util_trackmate.files_to_numpy(
        y_list, t_list, conversion
    )
    return label_true, label_trackmate


def detect_spots(input_image: np.ndarray) -> np.ndarray:
    """
    Detects spots in an image returning the coordinates and size.
    Returns in the format "row (y), column (x), sigma"
    """
    img = input_image
    img = ndi.filters.gaussian_filter(img, 2)
    img = skimage.exposure.equalize_hist(img, nbins=512)
    blobs = skimage.feature.blob_log(
        img, min_sigma=1.0, max_sigma=50, threshold=0.2, exclude_border=True
    )

    xy = np.stack([blobs[..., 1], blobs[..., 0]]).T
    return xy


def compute_score(
    true: np.ndarray, pred: np.ndarray, size: int, cell_size: int, weight: float
) -> pd.DataFrame:
    """Compute f1 score, error on coordinate and weighted f1/error"""
    f1_score = pd.Series(
        [
            training.util_metrics._f1_score(p, t, size, cell_size)
            for p, t in zip(true, pred)
        ]
    )

    err_coordinate = pd.Series(
        [
            training.util_metrics._error_on_coordinates(p, t, size, cell_size)
            for p, t in zip(true, pred)
        ]
    )

    weighted_f1_score_error_coordinates = pd.Series(
        [
            training.util_metrics._weighted_average_f1_score_error_coordinates(
                p, t, size, cell_size, weight
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
    parser.add_argument("-d", "--dataset", type=str, help="Path of the dataset folder.")
    parser.add_argument(
        "-t",
        "--trackmate",
        type=str,
        help="Path of directory containing labels and trackmate subfolders",
    )
    parser.add_argument(
        "-c",
        "--conversion",
        type=float,
        help="Rescaling factor to convert coordinates into pixel unit, required if --trackmate is defined",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=True,
        help="Size of images",
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

    size = args.size
    cell_size = args.cell_size
    weight = 1
    if args.weight is not None:
        weight = args.weight

    if args.trackmate is not None:
        print(f"Using trackmate data at {args.trackmate}")

        if args.conversion is None:
            raise ValueError("--trackmate requires --conversion.")

        trackmate = args.trackmate
        conversion = args.conversion
        
        label_true, label_trackmate = load_trackmate_data(trackmate, conversion)
        df = compute_score(
            true=label_true,
            pred=label_trackmate,
            size=size,
            cell_size=cell_size,
            weight=weight,
        )
        df_describe = df.describe()
        df_describe.to_csv(
            f"{os.path.splitext(args.trackmate)[0]}_trackmate_baseline.csv"
        )

    else:

        if args.dataset is None:
            raise ValueError("Provide either dataset or trackmate")

        print(f"Using dataset at {args.dataset}")

        with np.load(args.dataset, allow_pickle=True) as data:
            train_x = data["x_train"]
            valid_x = data["x_valid"]
            test_x = data["x_test"]
            train_y = data["y_train"]
            valid_y = data["y_valid"]
            test_y = data["y_test"]

        train_pred = list(map(detect_spots, train_x))
        valid_pred = list(map(detect_spots, valid_x))
        test_pred = list(map(detect_spots, test_x))

        for true, pred, name in zip(
            [train_y, valid_y, test_y], [train_pred, valid_pred, test_pred], ["train","valid","test"]
        ):
            df = compute_score(
                true=true,
                pred=pred,
                size=size,
                cell_size=cell_size,
                weight=weight,
            )
            df_describe = df.describe()
            df_describe.to_csv(f"{os.path.splitext(args.dataset)[0]}.{name}_baseline.csv")


if __name__ == "__main__":
    main()
