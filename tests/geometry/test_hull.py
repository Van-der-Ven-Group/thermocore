import pytest
import numpy as np
from thermocore.geometry.hull import (
    barycentric_coordinates,
    hull_distance_correlations,
    lower_hull,
    simplex_energy_equation_matrix,
    lower_hull_simplex_containing,
)
from tests.geometry.binary import (
    binary_points,
    binary_lower_hull_vertex_indices,
    binary_lower_hull_simplices,
    binary_hull,
    binary_lower_hull_simplex_indices,
    ZrN_FCC_composition_subset,
    ZrN_FCC_formation_energy_subset,
    ZrN_FCC_corr_subset,
)


def test_lower_hull_binary(
    binary_points,
    binary_hull,
    binary_lower_hull_vertex_indices,
    binary_lower_hull_simplices,
):
    """Tests lower_hull for binary data."""
    lower_hull_vertex_indices, lower_hull_simplex_indices = lower_hull(binary_hull)
    assert set(lower_hull_vertex_indices) == set(binary_lower_hull_vertex_indices)
    simplices_result_set = [
        frozenset(simplex)
        for simplex in binary_hull.simplices[lower_hull_simplex_indices]
    ]
    simplices_expected_set = [
        frozenset(simplex) for simplex in binary_lower_hull_simplices
    ]
    assert simplices_result_set == simplices_expected_set


def test_simplex_energy_equation_matrix_binary(
    binary_hull, binary_lower_hull_simplex_indices
):
    """Tests simplex_energy_equation_matrix for binary data, on lower hull simplices."""
    equation_matrix_result = simplex_energy_equation_matrix(
        binary_hull, binary_lower_hull_simplex_indices
    )
    equation_matrix_expected = np.array([[-7, 8], [-0.25, 1.25], [0.5, -2.5]])
    assert np.allclose(equation_matrix_result, equation_matrix_expected)


def test_energy_equation_matrix_binary_vertical(binary_hull):
    """Tests simplex_energy_equation_matrix for binary data, on all simplices.
    Should encounter error due to vertical simplex."""
    with pytest.raises(ValueError):
        simplex_energy_equation_matrix(binary_hull, range(len(binary_hull.simplices)))


def test_lower_hull_simplex_containing_binary(
    binary_hull, binary_lower_hull_simplex_indices
):
    """Tests lower_hull_simplex_containing for binary data."""
    test_points = np.array([0, 0.5, 1, 3, 5, 7.5, 9])
    possible_simplex_indices = [
        [binary_lower_hull_simplex_indices[i] for i in possible]
        for possible in [[0], [0], [0, 1], [1], [1, 2], [2], [2]]
    ]
    expected_energies = [8, 4.5, 1, 0.5, 0, 1.25, 2]
    result = lower_hull_simplex_containing(test_points, binary_hull)
    result_indices_provided = lower_hull_simplex_containing(
        test_points, binary_hull, binary_lower_hull_simplex_indices
    )
    assert np.array_equal(result[0], result_indices_provided[0])
    assert np.array_equal(result[1], result_indices_provided[1])
    for i in range(len(test_points)):
        assert result[0][i] in possible_simplex_indices[i]
        assert np.isclose(result[1][i], expected_energies[i])


def test_lower_hull_simplex_containing_binary_out_of_bounds(binary_hull):
    """Tests lower_hull_simplex_containing for binary data with points
    that are out of bounds, which should cause errors."""
    test_points_low = np.array([-0.5])
    with pytest.raises(ValueError):
        lower_hull_simplex_containing(test_points_low, binary_hull)
    test_points_high = np.array([9.5])
    with pytest.raises(ValueError):
        lower_hull_simplex_containing(test_points_high, binary_hull)


def test_barycentric_coordinates_1D():
    """Tests barycentric_coordinates for points in one dimension."""
    for a in np.linspace(-2, 2, 10):
        for d in np.linspace(1, 5, 10):
            b = a + d
            for p in np.linspace(a, b, 10):
                coordinates = barycentric_coordinates(
                    np.array([p]), np.array([[a], [b]])
                )
                x1 = 1 - (p - a) / (b - a)
                x2 = 1 - x1
                assert np.allclose(coordinates, np.array([x1, x2]))


def test_hull_correlation_calculator():
    """Tests the function 'hull_distance_correlations' by confirming that all points on the lower convex hull have hull correlations equal to the zero vector. """
    hullcorr = hull_distance_correlations(
        ZrN_FCC_corr_subset(),
        ZrN_FCC_composition_subset(),
        ZrN_FCC_formation_energy_subset(),
    )

    precalculated_lower_hull_indices = np.array([0, 1, 21, 22, 23, 24, 25, 26])
    assert np.allclose(hullcorr[precalculated_lower_hull_indices], 0)


# TODO: Tests for ternary data
