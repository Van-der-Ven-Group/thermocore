import pytest
import numpy as np
from scipy.spatial import ConvexHull


@pytest.fixture
def binary_points():
    return np.array(
        [
            [5, 5],
            [1, 3],
            [5, 0],
            [1, 8],
            [7, 7],
            [1, 1],
            [3, 8],
            [9, 7],
            [6, 6],
            [4, 3],
            [3, 9],
            [5, 9],
            [4, 2],
            [2, 9],
            [7, 3],
            [8, 4],
            [9, 2],
            [5, 7],
            [6, 4],
            [8, 8],
            [6, 5],
            [6, 9],
            [5, 2],
            [8, 3],
            [0, 8],
        ],
        dtype=float,
    )


@pytest.fixture
def binary_lower_hull_vertex_indices():
    return np.array([2, 5, 16, 24])


@pytest.fixture
def binary_lower_hull_simplices():
    return np.array([[24, 5], [5, 2], [2, 16]])


@pytest.fixture
def binary_hull(binary_points):
    return ConvexHull(binary_points)


@pytest.fixture
def binary_lower_hull_simplex_indices(binary_hull, binary_lower_hull_simplices):
    return np.array(
        [
            [frozenset(simplex) for simplex in binary_hull.simplices].index(
                frozenset(known_simplex)
            )
            for known_simplex in binary_lower_hull_simplices
        ]
    )
