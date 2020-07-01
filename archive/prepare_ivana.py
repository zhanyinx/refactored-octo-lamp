"""Prepares a dataset from an images/ labels/ folder structure."""

import secrets
import sys
from typing import List, Iterable

import numpy as np
import pandas as pd
import skimage.io

sys.path.append("../")
from spot_detection.io import train_valid_split, remove_zeros
from spot_detection.data import get_prediction_matrix, random_cropping
from prepare import (
    get_file_lists,
    _parse_args,
)


def group_to_numpy(img: str, label: str, cell_size: int) -> Iterable[np.ndarray]:
    """Reads files groups, sorts them and returns numpy arrays."""
    image = skimage.io.imread(img)
    image = image / np.max(image)  # normalisation

    df = pd.read_csv(label)

    if min(image.shape) < 512 or len(df) < 5:
        return 0, 0  # type: ignore[return-value]

    xy = np.stack([df["y"].to_numpy(), df["x"].to_numpy()]).T
    xy = get_prediction_matrix(xy, image.shape[0], cell_size, image.shape[1])

    return image, xy


def files_to_numpy(images: List[str], labels: List[str], cell_size: int, crop_size: int) -> Iterable[np.ndarray]:
    """Converts file lists into numpy arrays."""
    np_images = []
    np_labels = []

    for img, label_ in zip(images, labels):
        image, label = group_to_numpy(img, label_, cell_size)

        if isinstance(image, np.ndarray) and any(i > crop_size for i in image.shape):
            image, label = random_cropping(image, label, cell_size, crop_size)

        np_images.append(image)
        np_labels.append(label)

    np_images = remove_zeros(np_images)
    np_labels = remove_zeros(np_labels)

    return np_images, np_labels


def main():
    """Parse command-line argument and prepare dataset."""
    args = _parse_args()

    x_list, y_list = get_file_lists(args.path, format_image=args.image_format, format_label=args.label_format)
    x_trainval, x_test, y_trainval, y_test = train_valid_split(
        x_list=x_list, y_list=y_list, valid_split=args.test_split
    )
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x_list=x_trainval, y_list=y_trainval, valid_split=args.valid_split
    )

    x_train, y_train = files_to_numpy(x_train, y_train, args.cell_size, 512)
    x_valid, y_valid = files_to_numpy(x_valid, y_valid, args.cell_size, 512)
    x_test, y_test = files_to_numpy(x_test, y_test, args.cell_size, 512)

    print(f"All files*: {len(x_list)}")
    print(f"All files: {len(x_train) + len(x_valid) + len(x_test)}")
    print(f"  - Train: {len(x_train)}")
    print(f"  - Valid: {len(x_valid)}")
    print(f"  - Test: {len(x_test)}")

    fname = f"../data/{args.basename}_{secrets.token_hex(4)}.npz"
    np.savez_compressed(
        fname, x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test
    )


if __name__ == "__main__":
    main()
