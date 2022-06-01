from collections.abc import Sequence
import copy
import numpy
from math import isclose


def pull_ecis_from_json(eci_json: dict) -> list:
    """Returns a list of ECIs from a CASMv1.2.0 eci.json file. Can specify specific linear function indices of interest.

    Parameters
    ----------
    ecis_json : dict
        Dictionary of eci.json data in CASM format.
        Note: ECIs in CASM are not divided by the multiplicity.

    Returns
    -------
    ecis : list (float)
        Complete vector of ecis
    """
    # Returns eci value or 0 for each ECI
    return [
        cluster.get("eci", 0)
        for orbit in eci_json["orbits"]
        for cluster in orbit["cluster_functions"]
    ]


def pull_multiplicity_from_json(basis_json: dict) -> list:
    """Returns multiplicities from a CASMv1.2.0 basis.json or eci.json file. Can specify specific linear function indices of interest.

    Parameters
    ----------
    basis_json : dict
        Dictionary of basis.json or eci.json data in CASM format.

    Returns
    -------
    multiplicity : list (int)
        Complete vector of the multiplicity for each eci
    """
    # Returns multiplicity for each cluster
    return [
        orbit["mult"]
        for orbit in basis_json["orbits"]
        for cluster in orbit["cluster_functions"]
    ]


def append_ECIs_to_basis_data(
    ecis: numpy.ndarray, basis_data: dict, basis_function_indices: Sequence = None
) -> dict:
    """Appends ECI values to a dictionary of CASM basis set data, such that they can be properly loaded into CASM. This function will write the eci values as given in ecis except for those that are exactly zero. Any zero-ing out within a tolerance range should be done in pre- or post-processing.

    Parameters
    ----------
    ecis : numpy.ndarray of floats, shape (n_eci,)
    basis_data : dict
        CASM basis set data (e.g. loaded from a basis.json file).
    basis_function_indices : Sequence[int]
        Indices of the basis functions (linear_function_indices) corresponding to `ecis` that want to be turned on. Default = None indicating a non-sparse input of ecis.

    Returns
    -------
    dict
        CASM basis set data with ECI values appended (e.g. to be written to an eci.json file).
    """

    eci_json = copy.deepcopy(basis_data)  # this is by reference
    # adds eci value to corresponding spot in basis_data['orbits'][index]['cluster_functions']['eci']

    # convert ecis to dictionary for conciseness later
    if basis_function_indices is None:
        eci_dict = {i: ecis[i] for i in range(len(ecis))}
    else:
        # makes dictionary with keys being the basis_function_indices and values being the ecis to write
        eci_dict = {basis_function_indices[i]: ecis[i] for i in range(len(ecis))}

    # writing ecis to json
    for orbit in eci_json["orbits"]:
        for cluster in orbit["cluster_functions"]:
            # gets the value to be written (or zero if the corresponding index doesn't exist in eci_dict)
            eci_value = eci_dict.get(cluster["linear_function_index"], 0)
            # writes eci_value to the json if it's non-zero
            if eci_value != 0:
                cluster["eci"] = eci_value
    return eci_json


# TODO: Write test function for this
def zero_out_ecis_vector(eci_vector: list, tol=1e-10):
    """Returns a vector where values within a tolerance to 0 are set to 0.

    Parameters
    ----------
    eci_vector : list
        List of ECI values
    tol : float
        Tolerance for zeroing out the ecis

    Returns
    -------
    eci_vector : list
        List of ECI values with values within a tolerance to 0 set to 0.
    """
    # Iterate through dictionary and find all eci keys
    for i, eci in enumerate(eci_vector):
        if isclose(eci, 0, abs_tol=tol):
            eci_vector[i] = 0
    return eci_vector
