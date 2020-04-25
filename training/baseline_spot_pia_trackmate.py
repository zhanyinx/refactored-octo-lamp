"""
Computes the baseline value metrics of spot detection given trackmate labels.
Returns a csv file with all metrics.
Here different metrics are used: f1_score, error on xy, and combination of f1_score and error on xy
"""

import argparse
import numpy as np
import os.path
import pandas as pd
import sys
from typing import Tuple

sys.path.append("../")

import training.util_metrics
import training.util_trackmate


def load_trackmate_data(
    path: dir, size: int, cell_size: int, conversion: float = 1
) -> Tuple[np.ndarray]:
    """
    Returns two lists of np.ndarray of shape (n,n,3):
    containing p,x,y. One for trackmate labeled, one for human labeled

    Args:
        - path (str): path to directory containing
            trackmate and labels subdirectories.
        - size (int): size of the images labeled by 
            trackmate or human
        - cell_size (int): size of cell used to calculate
            F1 score, precision and recall  
        - conversion (float): scaling factor used to convert
            coordinates into pixel unit (default = 1, no
            conversion)
    """

    if not all(isinstance(i, int) for i in (size, cell_size)):
        raise TypeError(
            f"size and cell_size must be int, but are {type(size), type(cell_size)}."
        )

    x_list, y_list, t_list = training.util_trackmate.trackmate_get_file_lists(path)
    (
        images,
        label_true,
        label_trackmate,
    ) = training.util_trackmate.trackmate_files_to_numpy(
        x_list, y_list, t_list, conversion, size, cell_size
    )
    return images, label_true, label_trackmate


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

    print(f"Using trackmate data at {args.trackmate}")

    if args.conversion is None:
        raise ValueError("--trackmate requires --conversion.")

    trackmate = args.trackmate
    conversion = args.conversion

    images, label_true, label_trackmate = load_trackmate_data(
        path=trackmate, conversion=conversion, size=size, cell_size=cell_size
    )
    df = compute_score(
        true=label_true, pred=label_trackmate, cell_size=cell_size, weight=weight
    )
    df_describe = df.describe()
    df_describe.to_csv(
        f"{os.path.splitext(args.trackmate)[0]}_trackmate_baseline.csv"
    )

if __name__ == "__main__":
    main()
