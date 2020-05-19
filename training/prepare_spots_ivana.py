"""Prepares a dataset from an images/ labels/ folder structure."""

import secrets
import sys
from typing import List, Iterable

import numpy as np
import pandas as pd
import skimage.io

sys.path.append("../")
from training.util_prepare import (
    train_valid_split,
    get_prediction_matrix,
    _parse_args,
    get_file_lists,
    remove_zeros,
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


def random_cropping(image: np.ndarray, mask: np.ndarray, cell_size: int, crop_size: int = 256) -> np.ndarray:
    """Randomly crops an image and mask to size crop_size.

    Args:
        - image: Image to be cropped.
        - mask: Mask to be cropped.
        - cell_size: size of cell used to calculate F1 score
        - crop_size: Size to crop image and mask (both dimensions).

    Returns:
        - crop_image, crop_mask: Cropped image and mask
          respectively with shape (crop_size, crop_size).
    """
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(f"image, mask must be np.ndarray but is {type(image), type(mask)}.")
    if not all(isinstance(i, int) for i in [crop_size, cell_size]):
        raise TypeError(f"crop_size, cell_size must be an int but is {type(crop_size), type(cell_size)}.")
    if crop_size == 0:
        raise ValueError("crop_size must be larger than 0.")
    if not all(image.shape[i] >= crop_size for i in range(2)):
        raise ValueError("crop_size must be smaller than image_size.")
    if crop_size % cell_size > 0:
        raise ValueError("Crop size must be a multiple of cell_size.")

    start_dim = [0, 0]
    if image.shape[0] > crop_size:
        start_dim[0] = int(np.floor(np.random.randint(low=0, high=image.shape[0] - crop_size) / cell_size) * cell_size)
    if image.shape[1] > crop_size:
        start_dim[1] = int(np.floor(np.random.randint(low=0, high=image.shape[1] - crop_size) / cell_size) * cell_size)

    cropped_image = image[start_dim[0] : (start_dim[0] + crop_size), start_dim[1] : (start_dim[1] + crop_size)]
    cropped_mask = mask[
        int(start_dim[0] / cell_size) : int((start_dim[0] + crop_size) / cell_size),
        int(start_dim[1] / cell_size) : int((start_dim[1] + crop_size) / cell_size),
        :,
    ]

    return cropped_image, cropped_mask


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
