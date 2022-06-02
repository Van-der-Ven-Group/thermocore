from collections.abc import Sequence
from typing import Callable
import numpy as np


def select_basis_functions_by_orbit_filters(
    orbit_filters: Sequence, basis_data: dict
) -> list:
    """Returns indices of basis functions (within `basis_data`) that meet all criteria in `orbit_filters`.

    Parameters
    ----------
    orbit_filters : Sequence of functions
        Boolean functions operating on individual orbit data to use as selection criteria.
        Each function must take a dict of one orbits's data (e.g. one element of `basis_data`["orbits"])
        as its sole parameter return a bool indicating whether the orbit meets that particular criterion.
    basis_data : dict
        CASM basis set data (e.g. loaded from a basis.json file).

    Returns
    -------
    selected_indices : list[int]
        Linear function indices of basis functions that meet all of the selection criteria.

    Example usage
    -------------
    # Pass filters to select_basis_functions_by_orbit_filters
    selection_indices = select_basis_functions_by_orbit_filters([max_length_filter(4.5,2), max_length_filter(4.0,3)], basis_data)
    ------------------------------------------------------------------
    """
    selected_indices = []
    for orbit in basis_data["orbits"]:
        if all([selection(orbit) for selection in orbit_filters]):
            selected_indices.extend(
                [
                    cluster["linear_function_index"]
                    for cluster in orbit["cluster_functions"]
                ]
            )
    return selected_indices


def max_length_filter(cluster_size: int, max_length: float) -> Callable:
    """Returns function that takes orbit_data as input for the selection filtering. Returned function will return False if a given orbit of size cluster_size has maximum cluster size > max_length. Returns True otherwise.

    Parameters
    ----------
    cluster_size: int
        Size (number of points) of clusters to apply filter to.
    max_length : float
        Maximum allowed length for two points within cluster.

    Returns
    -------
    sub_max_length_filter : function
        Function takes 1 argument: a dictionary of orbit data. The function will return a bool based on the input criteria.
    """

    def sub_max_length_filter(orbit_data: dict) -> bool:
        """Sub function for max_length_filter.
        Returns False if the orbit in `orbit_data` has `cluster_size` points and a maximum distance between points greater than `maximum_length`, returns True otherwise.

        Parameters
        ----------
        orbit_data : dict
            CASM basis function data for a specific orbit.

        Returns
        -------
        bool
            Whether the orbit passes the maximum length filter.
        """
        return (
            orbit_data["prototype"]["max_length"] <= max_length
            and len(orbit_data["prototype"]["sites"]) == cluster_size
        ) or len(orbit_data["prototype"]["sites"]) != cluster_size

    return sub_max_length_filter


def min_length_filter(cluster_size: int, min_length: float) -> Callable:
    """Returns function that takes orbit_data as input for the selection filtering. Returned function will return False if a given orbit of size cluster_size has minimum cluster size < min_length. Returns True otherwise.

    Parameters
    ----------
    cluster_size: int
        Size (number of points) of clusters to apply filter to.
    min_length : float
        Minimum allowed length for two points within cluster.

    Returns
    -------
    sub_min_length_filter : function
        Function takes 1 argument: a dictionary of orbit data. The function will return a bool based on the input criteria.
    """

    def sub_min_length_filter(orbit_data: dict) -> bool:
        """Sub function for max_length_filter.
        Returns False if the orbit in `orbit_data` has `cluster_size` points and a minimum distance between points less than `min_length`, returns True otherwise.

        Parameters
        ----------
        orbit_data : dict
            CASM basis function data for a specific orbit.

        Returns
        -------
        bool
            Whether the orbit passes the minimum length filter.
        """
        return (
            orbit_data["prototype"]["min_length"] >= min_length
            and len(orbit_data["prototype"]["sites"]) == cluster_size
        ) or len(orbit_data["prototype"]["sites"]) != cluster_size

    return sub_min_length_filter
