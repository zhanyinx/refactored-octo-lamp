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
        help="Size of images, required if --trackmate is defined",
    )
    parser.add_argument(
        "-z",
        "--cell_size",
        type=int,
        help="Size of the cell in the grid, required if --trackmate is defined",
    )
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    if args.trackmate is not None:
        print(f"Using trackmate data at {args.trackmate}")

        if args.conversion is None:
            raise ValueError("--trackmate requires --conversion.")

        if args.size is None:
            raise ValueError("--trackmate requires --size.")

        if args.cell_size is None:
            raise ValueError("--trackmate requires --cell_size.")

        trackmate = args.trackmate
        conversion = args.conversion
        size = args.size
        cell_size = args.cell_size

        label_true, label_trackmate = load_trackmate_data(trackmate, conversion)

        f1_score = pd.Series(
            [
                training.util_metrics._f1_score(p, t, size, cell_size)
                for p, t in zip(label_true, label_trackmate)
            ]
        )
        err_coordinate = pd.Series(
            [
                training.util_metrics._error_on_coordinates(p, t, size, cell_size)
                for p, t in zip(label_true, label_trackmate)
            ]
        )

        weighted_f1_score_error_coordinates = pd.Series(
            [
                training.util_metrics._weighted_average_f1_score_error_coordinates(
                    p, t, size, cell_size
                )
                for p, t in zip(label_true, label_trackmate)
            ]
        )

        df = pd.DataFrame(
            [f1_score, err_coordinate, weighted_f1_score_error_coordinates]
        ).T
        df.columns = ["f1_score", "err_coordinate", "weighted_average"]
        df_describe = df.describe()
        df_describe.to_csv(
            f"{os.path.splitext(args.trackmate)[0]}_trackmate_baseline.csv"
        )


if __name__ == "__main__":
    main()
