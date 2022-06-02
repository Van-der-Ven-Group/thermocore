import pytest
from thermocore.io.casm import casm_query_reader


@pytest.fixture
def query_data():
    query_dat = [
        {
            "name": "SCEL1",
            "comp": [[1.0]],
            "corr": [[1.0], [0.0], [0.0]],
            "formation_energy": 0.00,
            "arbitrary_key": 1,
        },
        {
            "name": "SCEL2",
            "comp": [[0.0]],
            "corr": [[1.0], [1.0], [1.0]],
            "formation_energy": 0.00,
            "arbitrary_key": 2,
        },
        {
            "name": "SCEL3",
            "comp": [[0.5]],
            "corr": [[0.16], [0.5], [0.2]],
            "formation_energy": -0.3,
            "arbitrary_key": 3,
        },
    ]
    return query_dat


@pytest.fixture
def expected_query_data():
    expected_data = {
        "name": ["SCEL1", "SCEL2", "SCEL3"],
        "comp": [[1.0], [0.0], [0.5]],
        "corr": [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.16, 0.5, 0.2]],
        "formation_energy": [0.0, 0.0, -0.3],
        "arbitrary_key": [1, 2, 3],
    }
    return expected_data


# test thermocore.io.casm.casm_query_reader
def test_query_reader(query_data, expected_query_data):
    """Compare results of casm_query_reader to expected output."""
    query_reader_test = casm_query_reader(query_data)
    assert query_reader_test == expected_query_data
