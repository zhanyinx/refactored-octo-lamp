"""
Prepares a dataset from an folder containing .tif images and .csv labels structure.
As usually generated synthetically in Fiji.
"""

from training.util_prepare import extract_basename, train_valid_split
from typing import List, Tuple
import argparse
import glob
import itertools
import numpy as np
import operator
import os
import pandas as pd
import secrets
import skimage.io
import sys
sys.path.append("../")


def get_file_lists(path: dir) -> Tuple[List[dir]]:
    """
    Extracts file paths and checks if respective files are present.

    Args:
        - Path: Relative or absolute location of directory containing
            images and masks subdirectories.

    Returns:
        - x_list, y_list: Lists of absolute file paths for the files found
            in the images or masks subdirectories.
    """
    if not os.path.exists(path):
        raise OSError(f"Path {path} must exist.")

    x_list = sorted(glob.glob(f"{path}*.tif"))
    y_list = sorted(glob.glob(f"{path}*.csv"))

    if not len(x_list) == len(y_list):
        raise ValueError(
            f"Length of images/ and labels/ must match: {len(x_list)} != {len(y_list)}.")
    if len(x_list) == 0:
        raise ValueError(f"No files found in path {path}.")

    x_basenames = [extract_basename(f) for f in x_list]
    y_basenames = [extract_basename(f) for f in y_list]

    if not all((x == y) for x, y in zip(x_basenames, y_basenames)):
        raise ValueError(f"Names of images/ and labels/ files must match.")

    x_list = [os.path.abspath(f) for f in x_list]
    y_list = [os.path.abspath(f) for f in y_list]

    return x_list, y_list


def import_data(x_list: List[dir], y_list: List[dir]) -> Tuple[np.ndarray, pd.DataFrame]:
    """ Opens files from lists as images and DataFrames. """

    images, labels = [], []
    n_images = 0
    for x, y in zip(x_list, y_list):

        # Image import
        image = skimage.io.imread(x)
        assert image.ndim == 3
        images.append(image)

        # Label import
        df = pd.read_csv(y)
        for _, row in df.iterrows():
            labels.append([
                row["t [sec]"] + n_images,
                row["y [pixel]"],
                row["x [pixel]"]
            ])
        n_images += min(image.shape)

    images = np.concatenate(images)
    df = pd.DataFrame(labels, columns=["img_index", "x", "y"])

    return images, df


def get_prediction_matrix(x_coords: List[float],
                          y_coords: List[float],
                          size_image: int = 512,
                          size_grid: int = 64) -> np.ndarray:
    """
    Returns a matrix containing information on whether a cell inside the
    image-wide grid contains 1+ spots or not.
    Additionally, the relative spot coordinates (x, y) are in the second and
    third z-dimension respectively.

    Args:
        - x_coords: List of all x-coordinates.
        - y_coords: List of matching y-coordinates.
        - size_image: Default image size to lay the grid on.
        - size_grid: Number of grid cells to be used.
    Returns:
        - matrix: Matrix representation of spot coordinates.
    """

    size_gridcell = size_image // size_grid
    matrix = np.zeros((size_grid, size_grid, 3))

    # Top left coordinates of every cell
    grid = np.array([x * size_gridcell for x in range(size_grid)])

    # TODO use np.where instead.
    for x, y in itertools.product(range(size_grid), range(size_grid)):

        grid_x = grid[x]
        grid_y = grid[y]

        curr_select = ((x_coords >= grid_x) &
                       (x_coords < grid_x + size_gridcell) &
                       (y_coords >= grid_y) &
                       (y_coords < grid_y + size_gridcell))

        spot_x = x_coords[curr_select]
        spot_y = y_coords[curr_select]
        assert len(spot_x) == len(spot_y)

        if spot_x:
            coord_rel = get_relative_coordinates(
                coord_spot=(spot_x, spot_y),
                coord_cell=(grid_x, grid_y),
                size_gridcell=size_gridcell)
            matrix[x, y, 0] = 1
            matrix[x, y, 1] = coord_rel[0]
            matrix[x, y, 2] = coord_rel[1]

    return matrix


def get_coordinate_list(matrix: np.ndarray,
                        size_image: int = 512,
                        size_grid: int = 64) -> np.ndarray:
    """
    Converts the prediction matrix into a list of coordinates.

    Note - if plotting with plt.scatter, x and y must be reversed!

    Args:
        - matrix: Matrix representation of spot coordinates.
        - size_image: Default image size the grid was layed on.
        - size_grid: Number of grid cells used.
    Returns:
        - Array of x, y coordinates with the shape (n, 2).
    """

    size_gridcell = size_image // size_grid
    coords_x = []
    coords_y = []

    # Top left coordinates of every cell
    grid = np.array([x * size_gridcell for x in range(size_grid)])

    # TODO use np.where instead.
    for x, y in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[x, y, 0] > 0.5:
            grid_x = grid[x]
            grid_y = grid[y]
            spot_x = matrix[x, y, 1]
            spot_y = matrix[x, y, 2]

            coord_abs = get_absolute_coordinates(
                coord_spot=(spot_x, spot_y),
                coord_cell=(grid_x, grid_y),
                size_gridcell=size_gridcell)

            coords_x.append(coord_abs[0])
            coords_y.append(coord_abs[1])

    return np.array([coords_x, coords_y])


def get_relative_coordinates(coord_spot: Tuple[float],
                             coord_cell: Tuple[float],
                             size_gridcell: int = 8) -> Tuple[float]:
    """
    Returns the relative cell coordinates from absolute image coordinates.
    """
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_abs = tuple(map(operator.sub, coord_spot, coord_cell))
    coord_rel = tuple(map(lambda x: x/size_gridcell, coord_abs))
    return coord_rel


def get_absolute_coordinates(coord_spot: Tuple[float],
                             coord_cell: Tuple[float],
                             size_gridcell: int = 8) -> Tuple[float]:
    """
    Returns the absolute image coordinates from relative cell coordinates.
    """
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_rel = tuple(map(lambda x: x*size_gridcell, coord_spot))
    coord_abs = tuple(map(operator.add, coord_cell, coord_rel))
    # coord_abs = tuple(map(lambda x: int(x), coord_abs))
    return coord_abs


def files_to_numpy(images: List[dir], labels: List[dir], cell_size: int = 4) -> Tuple[np.ndarray]:
    """ Converts file lists into numpy arrays. """
    np_images, labels = import_data(images, labels)

    np_labels = []
    for i in labels['img_index'].unique():
        curr_df = labels[labels['img_index'] == i].reset_index(drop=True)
        x = curr_df['x'].to_numpy()
        y = curr_df['y'].to_numpy()
        np_labels.append(get_prediction_matrix(x, y, size_grid=512//cell_size))

    np_images /= 255
    np_labels = np.array(np_labels)

    return np_images, np_labels


def _parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str,
                        help="Path of the dataset folder.")
    parser.add_argument("-b", "--basename", type=str, default="ds", required=False,
                        help="Basename of dataset.")
    parser.add_argument("-s", "--cell_size", type=int, default=4, required=False,
                        help="Size of the prediction cells.")
    parser.add_argument("-t", "--test_split", type=int, default=0.1, required=False,
                        help="Percentage split of test set.")
    parser.add_argument("-v", "--valid_split", type=int, default=0.2, required=False,
                        help="Precentage split of validation set.")
    args = parser.parse_args()

    return args


def main():
    """ Parse command-line argument and prepare dataset. """
    args = _parse_args()

    x_list, y_list = get_file_lists(args.path)
    x_trainval, x_test, y_trainval, y_test = train_valid_split(
        x_list=x_list, y_list=y_list, valid_split=args.test_split)
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x_list=x_trainval, y_list=y_trainval, valid_split=args.valid_split)

    x_train, y_train = files_to_numpy(x_train, y_train, args.cell_size)
    x_valid, y_valid = files_to_numpy(x_valid, y_valid, args.cell_size)
    x_test, y_test = files_to_numpy(x_test, y_test, args.cell_size)

    print(f"All files*: {len(x_list)}")
    print(f"All files: {len(x_train) + len(x_valid) + len(x_test)}")
    print(f"  - Train: {len(x_train)}")
    print(f"  - Valid: {len(x_valid)}")
    print(f"  - Test: {len(x_test)}")

    fname = f"{args.basename}_{secrets.token_hex(4)}.npz"
    np.savez_compressed(
        fname,
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test
    )


if __name__ == "__main__":
    main()
