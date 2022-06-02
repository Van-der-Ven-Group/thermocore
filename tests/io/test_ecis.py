import json
import pytest
import numpy as np
from thermocore.io import casm


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
def test_ecis_as_np_array():
    return np.array(
        [
            0.1,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            -1.2,
            -1.3,
            -1.4,
            -1.5,
            -1.6,
            -1.7,
            -1.8,
            -1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
        ]
    )


@pytest.fixture
def test_ecis_as_list():
    return [
        0.1,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        -1.2,
        -1.3,
        -1.4,
        -1.5,
        -1.6,
        -1.7,
        -1.8,
        -1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
    ]


@pytest.fixture
def test_basis_data():
    fname = "./data/HfNO_no_energy_data/fake_ecis_for_testing.json"
    with open(fname, "r") as f:
        return json.load(f)


@pytest.fixture
def test_eci_json():
    fname = "./data/HfNO_no_energy_data/fake_ecis_for_testing.json"
    with open(fname, "r") as f:
        return json.load(f)


@pytest.fixture
def multiplicities():
    return [
        1,
        1,
        1,
        1,
        4,
        4,
        6,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]


@pytest.fixture
def multiplicities_with_zeros():
    return [
        1,
        1,
        1,
        1,
        4,
        4,
        6,
        6,
        6,
        6,
        12,
        12,
        6,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        24,
        24,
        24,
        24,
        24,
        24,
        6,
        6,
        6,
        12,
        12,
        12,
        12,
        12,
        6,
        6,
    ]


@pytest.fixture
def test_ecis_as_list_with_zeros():
    return [
        0.1,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0,
        0,
        0,
        0,
        0,
        0,
        -1.2,
        -1.3,
        -1.4,
        -1.5,
        -1.6,
        -1.7,
        -1.8,
        -1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_eci_writing(
    test_basis_data,
    selected_indices,
    test_eci_json,
    test_ecis_as_np_array,
    test_ecis_as_list,
):
    """
    Test that the eci writer properly writes the ecis to the selected indices
    """
    # Test using np array for ecis
    assert (
        casm.append_ECIs_to_basis_data(
            test_ecis_as_np_array, test_basis_data, selected_indices
        )
        == test_eci_json
    )

    # Test using list for ecis
    assert (
        casm.append_ECIs_to_basis_data(
            test_ecis_as_list, test_basis_data, selected_indices
        )
        == test_eci_json
    )


def test_pull_eci_from_json(test_eci_json, test_ecis_as_list_with_zeros):

    assert casm.pull_ecis_from_json(test_eci_json) == test_ecis_as_list_with_zeros


def test_pull_multiplicity_from_json(test_eci_json, multiplicities_with_zeros):
    assert casm.pull_multiplicity_from_json(test_eci_json) == multiplicities_with_zeros
