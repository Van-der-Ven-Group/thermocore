from __future__ import annotations
import numpy as np


def regroup_query_by_config_property(casm_query_json_data: list) -> dict:
    """Groups CASM query data by property instead of by configuration. 

    Parameters:
    -----------
    casm_query_json_data: list
        List of dictionaries read from casm query json file.

    Returns:
    --------
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
