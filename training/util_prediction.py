# List of functions to convert prediction to x,y format for plotting

import numpy as np
import itertools
from typing import Tuple
import operator


def get_coordinate_list(matrix: np.ndarray, size_image: int = 512, size_grid: int = 64) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    Note - if plotting with plt.scatter, x and y must be reversed!

    Args:
        matrix: Matrix representation of spot coordinates.
        size_image: Default image size the grid was layed on.
        size_grid: Number of grid cells used.
    Returns:
        Array of x, y coordinates with the shape (n, 2).
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
                coord_spot=(spot_x, spot_y), coord_cell=(grid_x, grid_y), size_gridcell=size_gridcell,
            )

            coords_x.append(coord_abs[0])
            coords_y.append(coord_abs[1])

    return np.array([coords_x, coords_y]).T


def get_relative_coordinates(
    coord_spot: Tuple[float], coord_cell: Tuple[float], size_gridcell: int = 8
) -> Tuple[float]:
    """Return the relative cell coordinates from absolute image coordinates."""
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_abs = tuple(map(operator.sub, coord_spot, coord_cell))
    coord_rel = tuple(map(lambda x: x / size_gridcell, coord_abs))
    return coord_rel


def get_absolute_coordinates(
    coord_spot: Tuple[float], coord_cell: Tuple[float], size_gridcell: int = 8
) -> Tuple[float]:
    """Return the absolute image coordinates from relative cell coordinates."""
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_rel = tuple(map(lambda x: x * size_gridcell, coord_spot))
    coord_abs = tuple(map(operator.add, coord_cell, coord_rel))
    # coord_abs = tuple(map(lambda x: int(x), coord_abs))
    return coord_abs
