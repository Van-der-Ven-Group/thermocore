import json
import pytest
import numpy as np
from thermocore.io import casm


@pytest.fixture
def test_vector_as_list():
    return [
        0.1,
        1e-5,
        1e-4,
        100,
        1e-9,
        3.20e-8,
        -1e-9,
        -1e-14,
        1.1e-10,
        0.9e-10,
        1e-10,
        1e-11,
        -1e-10,
        -1e-11,
        -1e-15,
        2,
        1.0000000001e-10,
    ]


@pytest.fixture
def zeroed_vector_as_list_default():
    return [
        0.1,
        1e-5,
        1e-4,
        100,
        1e-9,
        3.20e-8,
        -1e-9,
        0,
        1.1e-10,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        1.0000000001e-10,
    ]


@pytest.fixture
def zeroed_vector_as_list_minus14():
    return [
        0.1,
        1e-5,
        1e-4,
        100,
        1e-9,
        3.20e-8,
        -1e-9,
        0,
        1.1e-10,
        0.9e-10,
        1e-10,
        1e-11,
        -1e-10,
        -1e-11,
        0,
        2,
        1.0000000001e-10,
    ]


def test_zero_vector(test_vector_as_list, zeroed_vector_as_list_default):
    assert casm.zero_out_vector(test_vector_as_list) == zeroed_vector_as_list_default


def test_zero_vector_minus14(test_vector_as_list, zeroed_vector_as_list_minus14):
    assert (
        casm.zero_out_vector(test_vector_as_list, tol=1e-14)
        == zeroed_vector_as_list_minus14
    )
