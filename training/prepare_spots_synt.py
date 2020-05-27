"""Prepares a dataset from an folder containing images and labels subfolders."""

import secrets
import sys
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
import skimage.io


sys.path.append("../")
from training.util_prepare import (
    train_valid_split,
    get_prediction_matrix,
    get_file_lists,
    _parse_args,
)


def import_data(x_list: List[str], y_list: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Opens files from lists as images and DataFrames."""
    n_images = 0
    images, labels = [], []
    for x, y in zip(x_list, y_list):

        # Image import
        image = skimage.io.imread(x)
        image = image / np.max(image)

        assert image.ndim == 3
        images.append(image)

        # Label import
        df = pd.read_csv(y)
        for _, row in df.iterrows():
            labels.append([row["t [sec]"] + n_images, row["y [pixel]"], row["x [pixel]"]])
        n_images += min(image.shape)

    images = np.concatenate(images)
    df = pd.DataFrame(labels, columns=["img_index", "x", "y"])

    return images, df


def files_to_numpy(images_: List[str], labels_: List[str], cell_size: int = 4) -> Iterable[np.ndarray]:
    """Converts file lists into numpy arrays."""
    np_images, labels = import_data(images_, labels_)

    np_labels = []
    for i in labels["img_index"].unique():
        curr_df = labels[labels["img_index"] == i].reset_index(drop=True)
        xy = np.stack([curr_df["x"].to_numpy(), curr_df["y"].to_numpy()]).T
        np_labels.append(get_prediction_matrix(xy, 512, cell_size))

    np_labels = np.array(np_labels)

    return np_images, np_labels


def main():
    """Parse command-line argument and prepare dataset."""
    args = _parse_args()

    x_list, y_list = get_file_lists(path=args.path, format_image=args.image_format, format_label=args.label_format)
    x_trainval, x_test, y_trainval, y_test = train_valid_split(
        x_list=x_list, y_list=y_list, valid_split=args.test_split
    )
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x_list=x_trainval, y_list=y_trainval, valid_split=args.valid_split
    )

    x_train, y_train = files_to_numpy(x_train, y_train, args.cell_size)
    x_valid, y_valid = files_to_numpy(x_valid, y_valid, args.cell_size)
    x_test, y_test = files_to_numpy(x_test, y_test, args.cell_size)

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
