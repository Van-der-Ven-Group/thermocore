from __future__ import annotations
import numpy as np


def casm_query_reader(casm_query_json_data: list) -> dict:
    """Reads keys and values from casm query json dictionary.

    Parameters:
    -----------
    casm_query_json_data: list
        List of dictionaries read from casm query json file.

    Returns:
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)

    Notes:
    ------
    Casm query jsons are lists of dictionaries; each dictionary corresponds to a configuration. 
    This function assumes that all dictionaries have the same keys. It sorts all properties by those keys instead of by configuration.
    Properties that are a single value or string are passed as a list of those properties. 
    Properties that are arrays are passed as a list of lists (2D matrices) even if the property only has one value (a matrix of one column). 
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
        comp = np.array(results["comp"])
        if len(comp.shape) > 2:
            results["comp"] = np.squeeze(comp).tolist()
    if "corr" in results.keys():
        results["corr"] = np.squeeze(results["corr"]).tolist()
    return results


def write_eci_dict(eci: numpy.ndarray, basis_json_dict: dict) -> dict:
    """Writes supplied ECI to the dictionary found in basis.json. This can then be written to an eci.json file.

    Parameters:
    -----------
    eci: numpy.ndarray
        Vector of ECI values.

    basis_json_dict: dict
        dictionary read directly from basis.json file.

    Returns:
    --------
    data: dict
        basis.json dictionary formatted with provided eci's
    """

    for index, orbit in enumerate(basis_json_dict["orbits"]):
        basis_json_dict["orbits"][index]["cluster_functions"]["eci"] = eci[index]

    return basis_json_dict


def label_missing_energies(energies: numpy.ndarray) -> numpy.ndarray:
    """Labels missing energies with 'np.nan'. 

    Parameters:
    -----------
    energies: np.ndarray
        Array of energies, including missing energies.

    Returns:
    --------
    energies: np.ndarray
        Array of energies with missing energies labeled with 'np.nan'
    """

    # Casm 1.2.0 uses None, casm 0.3 uses {}
    uncalculated_energy_descriptor = None
    if {} in energies:
        uncalculated_energy_descriptor = {}

    uncalculated_indices = np.where(
        (energies == uncalculated_energy_descriptor) == True
    )
    energies[uncalculated_indices] = np.nan
    return energies

