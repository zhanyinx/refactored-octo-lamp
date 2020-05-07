"""Baseline functions."""

import pandas as pd
import numpy as np

import training.util_metrics

def compute_score(true: np.ndarray, pred: np.ndarray, cell_size: int, weight: float) -> pd.DataFrame:
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
    f1_score = pd.Series([training.util_metrics.f1_score_(p, t) for p, t in zip(true, pred)])

    err_coordinate = pd.Series(
        [training.util_metrics.error_on_coordinates_(p, t, cell_size) for p, t in zip(true, pred)]
    )

    weighted_f1_score_error_coordinates = pd.Series(
        [
            training.util_metrics.weighted_average_f1_score_error_coordinates_(p, t, cell_size, weight)
            for p, t in zip(true, pred)
        ]
    )
    df = pd.DataFrame([f1_score, err_coordinate, weighted_f1_score_error_coordinates]).T
    df.columns = ["f1_score", "err_coordinate", "weighted_average"]
    return df
