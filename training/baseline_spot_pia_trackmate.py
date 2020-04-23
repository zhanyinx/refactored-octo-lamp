"""
Computes the baseline value metrics of nuclear segmentation given a dataset.
Returns a csv file with all metrics.
Here the dice metric is used.
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
sys.path.append("../")

from training.util_metrics import dice_coef


def nuclear_segmentation(input_image: np.ndarray) -> np.ndarray:
    """
    Detects and segments nuclear instances.

    Args:
        - image: Original image to be segmented.
    Returns:
        - image: Segmented images with labeled nuclei.
    """
    if not isinstance(input_image, np.ndarray):
        raise TypeError(
            f"input_image must be np.ndarray but is {type(input_image)}.")

    image = ndi.gaussian_filter(input_image, 5)
    image = image > skimage.filters.threshold_otsu(image)
    otsu = image
    image = ndi.binary_erosion(image)
    image = ndi.distance_transform_edt(image)
    image = image > image.mean()
    lab = ndi.label(image)[0]
    image = skimage.morphology.watershed(input_image, lab, mask=otsu)
    return image


def _parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        help="Path of the dataset folder.")
    parser.add_argument("-t", "--trackmate", type=str,
                        help="Path of the trackmate coordinates folder.")
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    with np.load(args.dataset) as data:
        train_x = data["x_train"]
        valid_x = data["x_valid"]
        test_x = data["x_test"]
        train_y = data["y_train"]
        valid_y = data["y_valid"]
        test_y = data["y_test"]

    train_pred = list(map(nuclear_segmentation, train_x))
    valid_pred = list(map(nuclear_segmentation, valid_x))
    test_pred = list(map(nuclear_segmentation, test_x))

    train_dice = pd.Series([dice_coef(p, t)
                            for p, t in zip(train_pred, train_y)])
    valid_dice = pd.Series([dice_coef(p, t)
                            for p, t in zip(valid_pred, valid_y)])
    test_dice = pd.Series([dice_coef(p, t)
                           for p, t in zip(test_pred, test_y)])

    df = pd.DataFrame([train_dice, valid_dice, test_dice]).T
    df.columns = ["train", "valid", "test"]
    df_describe = df.describe()
    df_describe.to_csv(f"{os.path.splitext(args.dataset)[0]}_baseline.csv")


if __name__ == "__main__":
    main()
