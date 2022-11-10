from __future__ import annotations
import numpy as np
from collections.abc import Sequence
import copy


def regroup_query_by_config_property(casm_query_json_data: list) -> dict:
    """Groups CASM query data by property instead of by configuration.

    Parameters
    ----------
    casm_query_json_data: list
        List of dictionaries read from casm query json file.

    Returns
    -------
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)

    Notes
    ------
    Casm query jsons are lists of dictionaries; each dictionary corresponds to a configuration.
    This function assumes that all dictionaries have the same keys.
    It sorts all properties by those keys instead of by configuration.
    Properties that are a single value or string are passed as a list of those properties.
    Properties that are arrays are passed as a list of lists (2D matrices) even if the
    property only has one value (a matrix of one column).
    """
    data = casm_query_json_data
    keys = data[0].keys()
    data_collect = []
    for i in range(len(keys)):
        data_collect.append([])

    for element_dict in data:
        for index, key in enumerate(keys):
            data_collect[index].append(element_dict[key])

    results = dict(zip(keys, data_collect))

    if "comp" in results.keys():
        # Enforce that composition is always rank 2.
        results["comp"] = np.array(results["comp"])
        if len(results["comp"].shape) > 2:
            results["comp"] = np.squeeze(results["comp"])
        if len(results["comp"].shape) == 1:
            results["comp"] = np.reshape(results["comp"], (-1, 1))
        results["comp"] = results["comp"].tolist()

    if "corr" in results.keys():
        # Remove redundant dimensions in correlation matrix.
        results["corr"] = np.squeeze(results["corr"]).tolist()
    return results


def pull_ecis_from_json(eci_json: dict) -> list:
    """Returns a list of ECIs from a CASMv1.2.0 eci.json file.
    Can specify specific linear function indices of interest.

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
    """Returns multiplicities from a CASMv1.2.0 basis.json or eci.json file.
    Can specify specific linear function indices of interest.

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
    ecis: np.ndarray, basis_data: dict, basis_function_indices: Sequence = None
) -> dict:
    """Appends ECI values to a dictionary of CASM basis set data, such that they can be properly loaded into CASM.
    This function will write the eci values as given in ecis except for those that are exactly zero.
    Any zero-ing out within a tolerance range should be done in pre- or post-processing.

    Parameters
    ----------
    ecis : numpy.ndarray of floats, shape (n_eci,)
    basis_data : dict
        CASM basis set data (e.g. loaded from a basis.json file).
    basis_function_indices : Sequence[int]
        Indices of the basis functions (linear_function_indices) corresponding to `ecis` that want to be turned on.
        Default = None indicating a non-sparse input of ecis.

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


def upscale_eci_vector(
    pruned_ecis: np.ndarray, eci_is_nonzero: np.ndarray
) -> np.ndarray:
    """Intended for cases when an ECI vector is pruned, and a user would like to then upscale the ECI vector back to its original size.
        Takes a pruned vector of nonzero ECIs of length m, and a mask of length n, where n is the length of the original (non-sparse) ECI vector, and n>m.
        The mask is a boolean array of length n, where the nonzero entries correspond to the indices of the ECIs that were pruned.
        The output is a vector of length n, where the nonzero entries come from the pruned_ecis vector, and the pruned entries are zero.


    Parameters
    ----------
    ecis : np.ndarray
        Vector of ECI values, shape (m,)
    eci_is_nonzero : np.ndarray
        Vector of Booleans, shape (n,) where True indicates the ECI is included in the vector. Number of True should equal the length of the ECI vector. Otherwise, the function will return an error.

    Returns
    -------
    ecis_upscaled : np.ndarray
        Vector of ECI values, shape (n,) upscaled to the original size of n.
    """
    if np.sum(eci_is_nonzero) != len(pruned_ecis):
        raise ValueError(
            "The number of True elements in the mask must equal the length of the ECI vector."
        )
    # Create a new vector of zeros. The length of the new vector will be the same as the mask.
    # Find the indices of the mask where the value is 1
    # Replace the zeros in the new vector with the ECI values, in the order of the indices of the mask where the value is 1
    indices = np.nonzero(eci_is_nonzero)[0]
    ecis_upscaled = np.zeros(len(eci_is_nonzero))
    ecis_upscaled[indices] = pruned_ecis
    return ecis_upscaled
