import json
import numpy as np


def casm_query_reader(casm_query_json_path="pass", casm_query_json_data=None):
    """Reads keys and values from casm query json dictionary.
    Parameters:
    -----------
    casm_query_json_path: str
        Absolute path to casm query json file.
        Defaults to 'pass' which means that the function will look to take a dictionary directly.
    casm_query_json_data: dict
        Can also directly take the casm query json dictionary.
        Default is None.

    Returns:
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)
    """
    if casm_query_json_data is None:
        with open(casm_query_json_path) as f:
            data = json.load(f)
    else:
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
