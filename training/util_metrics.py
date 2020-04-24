import numpy as np
from typing import List


def _create_spot_mask(spot_coord: np.ndarray, size: int, cell_size: int) -> np.ndarray:
    """Create mask image with spot"""
    img = np.zeros((size // cell_size, size // cell_size, 3))
    for i in range(len(spot_coord)):
        x = int(np.floor(spot_coord[i, 0])) // cell_size
        y = int(np.floor(spot_coord[i, 1])) // cell_size
        img[x, y, 0] = 1
        img[x, y, 1] = spot_coord[i, 0]
        img[x, y, 2] = spot_coord[i, 1]

    return img


def _euclidian_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidian distance between two points"""
    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return d


def _precision(pred: np.ndarray, true: np.ndarray, size: int, cell_size: int) -> float:
    """
    Calculate _precision (True positive)/ (True positive + False positive)
    """
    img_container_true = _create_spot_mask(true, size, cell_size)
    img_container_pred = _create_spot_mask(pred, size, cell_size)

    selection = img_container_pred[..., 0] == 1
    p = np.mean(img_container_true[selection, 0])
    return p


def _recall(pred: np.ndarray, true: np.ndarray, size: int, cell_size: int) -> float:
    """
    Calculate _recall (True positive) / (True positive / False negative)
    """
    img_container_true = _create_spot_mask(true, size, cell_size)
    img_container_pred = _create_spot_mask(pred, size, cell_size)

    selection = img_container_true[..., 0] == 1
    r = np.mean(img_container_pred[selection, 0])
    return r


def _f1_score(pred: np.ndarray, true: np.ndarray, size: int, cell_size: int) -> float:
    """Calculate f1 score  = 2*_precision*_recall/(_precision+_recall)"""
    r = _recall(pred, true, size, cell_size)
    p = _precision(pred, true, size, cell_size)
    
    if r==0 and p==0:
        return 0

    f1_score = 2 * p * r / (p + r)
    return f1_score


def _error_on_coordinates(
    pred: np.ndarray, true: np.ndarray, size: int, cell_size: int
) -> List[float]:
    """
    Calculate error on the coordinate of true positives
    """
    img_container_true = _create_spot_mask(true, size, cell_size)
    img_container_pred = _create_spot_mask(pred, size, cell_size)

    spot = (img_container_true[..., 0] == 1) & (img_container_pred[..., 0] == 1)
    d = 0
    counter = 0
    assert img_container_pred.shape == img_container_true.shape

    for i in range(len(img_container_pred)):
        for j in range(len(img_container_pred)):
            if spot[i, j]:
                x1 = img_container_true[i, j, 1]
                x2 = img_container_pred[i, j, 1]
                y1 = img_container_true[i, j, 2]
                y2 = img_container_pred[i, j, 2]
                d += _euclidian_dist(x1=x1, y1=y1, x2=x2, y2=y2)
                counter += 1

    if counter:
        d = d / counter
    else:
        d = None

    return d


def _weighted_average_f1_score_error_coordinates(
    pred: np.ndarray, true: np.ndarray, size: int, cell_size: int, weight: float = 1
) -> float:
    """Return (weight*(1-f1_score) + error_coordinates)/2"""
    f1_score = _f1_score(pred, true, size, cell_size)
    f1_score = 1 - f1_score
    f1_score = f1_score * weight

    error_coordinates = _error_on_coordinates(pred, true, size, cell_size)
    if error_coordinates is not None:
        score = (f1_score + error_coordinates) / 2
        return score

    return None
