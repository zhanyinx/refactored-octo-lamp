""" Prepares a dataset from an images/ labels/ folder structure."""

import secrets
import sys
from typing import List, Iterable

import skimage.io
import pandas as pd
import numpy as np

sys.path.append("../")
from training.util_prepare import train_valid_split, get_prediction_matrix, _parse_args, get_file_lists, remove_zeros


def group_to_numpy(img: str, label: str, conversion: float, cell_size: int, bitdepth: int) -> Iterable[np.ndarray]:
    """ Reads files groups, sorts them, convert coordinates to pixel unit and returns numpy arrays."""

    image = skimage.io.imread(img)
    image /= 2 ** bitdepth - 1  # normalise

    df = pd.read_table(label)
    if min(image.shape) < 512 or len(df) < 5:
        return 0, 0  # type: ignore[return-value]

    df.columns = ["x", "y"]
    xy = np.stack([df["y"].to_numpy(), df["x"].to_numpy()]).T
    xy = xy * conversion
    xy = get_prediction_matrix(xy, len(image), cell_size)

    return image, xy


def files_to_numpy(
    images: List[str], labels: List[str], conversion: float, cell_size: int, bitdepth: int,
) -> Iterable[np.ndarray]:
    """ Converts file lists into numpy arrays. """
    np_images = []
    np_labels = []

    for image, label in zip(images, labels):
        image, label = group_to_numpy(image, label, conversion, cell_size, bitdepth)
        np_images.append(image)
        np_labels.append(label)

    np_images = remove_zeros(np_images)
    np_labels = remove_zeros(np_labels)

    return np_images, np_labels


def main():
    """ Parse command-line argument and prepare dataset. """
    args = _parse_args()

    x_list, y_list = get_file_lists(args.path, format_image=args.image_format, format_label=args.label_format)
    x_trainval, x_test, y_trainval, y_test = train_valid_split(
        x_list=x_list, y_list=y_list, valid_split=args.test_split
    )
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x_list=x_trainval, y_list=y_trainval, valid_split=args.valid_split
    )

    x_train, y_train = files_to_numpy(x_train, y_train, args.conversion, args.cell_size, args.bitdepth)
    x_valid, y_valid = files_to_numpy(x_valid, y_valid, args.conversion, args.cell_size, args.bitdepth)
    x_test, y_test = files_to_numpy(x_test, y_test, args.conversion, args.cell_size, args.bitdepth)

    print(f"All files*: {len(x_list)}")
    print(f"All files: {len(x_train) + len(x_valid) + len(x_test)}")
    print(f"  - Train: {len(x_train)}")
    print(f"  - Valid: {len(x_valid)}")
    print(f"  - Test: {len(x_test)}")

    fname = f"../data/{args.basename}_{secrets.token_hex(4)}.npz"
    np.savez_compressed(
        fname, x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test,
    )


if __name__ == "__main__":
    main()
