import json
import pytest
import numpy as np
from thermocore.io import basis_selection


@pytest.fixture
def test_basis_data():
    fpath = "./tests/io/data/HfNO_no_energy_data/basis.json"
    with open(fpath, "r") as f:
        basis_data = json.load(f)
    return basis_data


@pytest.fixture
def selected_indices():
    # max_len: 0, point, pair<3, triple<3.5 (21 entries)
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]


@pytest.fixture
def selected_indices_2():
    # min_len: 0, point, pair>3, triple>3.5 (n entries)
    return [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]


def test_index_selection_via_max_length_filters(test_basis_data, selected_indices):
    """
    Test that the selected_indices from selection filter match the manually selected ones.
    Empty, point, pairs <3, triplets <3.5
    """

    test_selection = basis_selection.select_basis_functions_by_orbit_filters(
        [
            basis_selection.max_length_filter(2, 3),
            basis_selection.max_length_filter(3, 3.5),
        ],
        test_basis_data,
    )
    assert selected_indices == test_selection


def test_index_selection_via_min_length_filters(test_basis_data, selected_indices_2):
    """
    Test that the selected_indices from selection filter match the manually selected ones.
    Empty, point, pairs >3, triplets >3.5
    """

    test_selection = basis_selection.select_basis_functions_by_orbit_filters(
        [
            basis_selection.min_length_filter(2, 3),
            basis_selection.min_length_filter(3, 3.5),
        ],
        test_basis_data,
    )
    print(test_selection)
    assert selected_indices_2 == test_selection
