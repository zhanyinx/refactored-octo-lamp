import numpy as np
from typing import List


def _euclidian_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidian distance between two points"""
    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return d


def _precision(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Returns the precision defined as (True positive)/(True positive + False positive).
    Precision will be measured within cell of size "cell_size". If cell_size = 1, precision
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        - pred: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
    """

    selection = pred[..., 0] == 1
    p = np.mean(true[selection, 0])
    return p


def _recall(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Returns the recall defined as (True positive)/(True positive + False negative).
    Recall will be measured within cell of size "cell_size". If cell_size = 1, recall
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        - pred: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
    """

    selection = true[..., 0] == 1
    r = np.mean(pred[selection, 0])
    return r


def _f1_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Returns F1 score defined as: 
    2 * precision*recall / precision+recall

    F1 score will be measured within cell of size "cell_size". If cell_size = 1, F1 score
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        - pred: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
    """
    r = _recall(pred, true)
    p = _precision(pred, true)

    if r == 0 and p == 0:
        return 0

    f1_score = 2 * p * r / (p + r)
    return f1_score


def _error_on_coordinates(pred: np.ndarray, true: np.ndarray, cell_size: int) -> float:
    """
    Returns average error on spot coordinates.

    Args:
        - pred: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
        - cell_size (int): size of cell used to calculate
            F1 score, precision and recall   
    """

    spot = (true[..., 0] == 1) & (pred[..., 0] == 1)
    d = 0.0
    counter = 0
    assert pred.shape == true.shape

    for i in range(len(pred)):
        for j in range(len(pred)):
            if spot[i, j]:
                x1 = true[i, j, 1] * cell_size
                x2 = pred[i, j, 1] * cell_size
                y1 = true[i, j, 2] * cell_size
                y2 = pred[i, j, 2] * cell_size
                d += _euclidian_dist(x1=x1, y1=y1, x2=x2, y2=y2)
                counter += 1

    if counter:
        d = d / counter
    else:
        d = None  # type: ignore

    return d


def _weighted_average_f1_score_error_coordinates(
    pred: np.ndarray, true: np.ndarray, cell_size: int, weight: float = 1
) -> float:
    """
    Returns weighted single score defined as: 
    weight*(1-F1) + (error on coordinate)

    F1 score will be measured within cell of size "cell_size". If cell_size = 1, F1 score
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        - pred: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell.
        - true: np.ndarray of shape (n, n, 3):
            p, x, y format for each cell
        - cell_size: size of cells in the grid used to calculate
            F1 score, relative coordinates
        - weight: weight of 1-F1 score in the average
            default = 1
    """
    f1_score = _f1_score(pred, true)
    f1_score = 1.0 - f1_score
    f1_score = f1_score * weight

    error_coordinates = _error_on_coordinates(pred, true, cell_size)
    if error_coordinates is not None:
        score = (f1_score + error_coordinates) / 2
        return score

    return None
