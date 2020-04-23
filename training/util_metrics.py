import numpy as np


def dice_coef(pred: np.ndarray,
              true: np.ndarray,
              empty_score: float = 1.0) -> float:
    """
    Pure numpy implementation to calculate the Dice coefficient,
    a measure of set similarity.
    It is calculated as follows:

    DICE = 2*TP / (2*TP + FP + FN)

    The order of inputs for `dice` is irrelevant. The result will be
    identical if `pred` and `true` are switched.

    Args:
        - pred: Array of arbitrary size. If not boolean, will be converted.
        - true: An other array of identical size. If not boolean, will be converted.
    Returns:
        - dice: Dice coefficient as a float on range [0,1].
            Maximum similarity = 1, No similarity = 0, 
            Both are empty (sum eq to zero) = empty_score.

    Adapted from @brunodoamaral.
    """
    if not all(isinstance(i, np.ndarray) for i in [pred, true]):
        raise TypeError(
            f"pred, true must be np.ndarray but are {type(pred), type(true)}.")
    if not isinstance(empty_score, float):
        raise TypeError(
            f"empty_score must be float but is {type(empty_score)}.")

    pred = np.asarray(pred).astype(np.bool)
    true = np.asarray(true).astype(np.bool)

    if pred.shape != true.shape:
        raise ValueError(
            f"Pred and true must match in shape: {pred.shape} != {true.shape}.")

    img_sum = pred.sum() + true.sum()
    if img_sum == 0:
        return empty_score

    intersection = np.logical_and(pred, true)

    return 2.*intersection.sum() / img_sum
