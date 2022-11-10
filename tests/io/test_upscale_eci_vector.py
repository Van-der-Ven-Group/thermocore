import pytest
from thermocore.io.casm import upscale_eci_vector
import numpy as np


def test_upscale_eci_vector():
    un_pruned_vector = np.array([1, 6, 0, 3, 5, 0, 5, 8, 0.0, 2])
    pruned_vector = np.array([1, 6, 3, 5, 5, 8, 2])
    eci_is_nonzero = ~np.isclose(un_pruned_vector, 0)

    assert np.allclose(
        upscale_eci_vector(pruned_vector, eci_is_nonzero), un_pruned_vector
    )
