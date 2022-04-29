import pytest
import numpy as np
from thermocore.io.casm import label_missing_energies


@pytest.fixture
def incomplete_energies():
    return np.array([-0.1, None, -0.6, None, -0.2])


def test_filter(energies=incomplete_energies()):
    """Tests calculated_filter function from thermofitting.analysis.posterior_analysis
    """
    assert np.all(
        np.array([-0.1, np.nan, -0.6, np.nan, -0.2]) == label_missing_energies(energies)
    )
