from collections.abc import Sequence
from typing import Callable


def select_basis_functions_by_orbit_filters(
    orbit_filters: Sequence, basis_data: dict
) -> list:
    """Returns indices of basis functions (within `basis_data`) that meet all criteria in `orbit_filters`.

    Parameters
    ----------
    orbit_filters : Sequence of functions
        Boolean functions operating on individual orbit data to use as selection criteria.
        Each function must take a dict of one orbits's data (e.g. one element of `basis_data`["orbits"])
        as its sole parameter and returns a bool indicating whether the orbit meets that particular criterion.
            If the orbit does not meet the criterion, the orbit is excluded from the selection.
            If the orbit cannot be filtered by the function (ie. different cluster_size for length filters),
            the filter returns None and is ignored.
    basis_data : dict
        CASM basis set data (e.g. loaded from a basis.json file).

    Returns
    -------
    selected_indices : list[int]
        Linear function indices of basis functions that meet all of the selection criteria (ignores None).

    Examples
    --------
    Pass filters to select_basis_functions_by_orbit_filters
        selection_indices =
            select_basis_functions_by_orbit_filters(
                    [max_length_filter(4.5,2),
                    max_length_filter(4.0,3)],
                    basis_data)

    """
    selected_indices = []
    for orbit in basis_data["orbits"]:
        if all(
            [
                select_orbit
                for select_orbit in [selection(orbit) for selection in orbit_filters]
                if select_orbit is not None
            ]
        ):
            selected_indices.extend(
                [
                    cluster["linear_function_index"]
                    for cluster in orbit["cluster_functions"]
                ]
            )
    return selected_indices


def max_length_filter(cluster_size: int, max_length: float) -> Callable:
    """Returns a function that takes orbit_data as input for the selection filtering.
    The returned function returns False if an orbit of size cluster_size
        has maximum cluster size > max_length.
    Function returns True if the orbit of size cluster_size
        has maximum cluster size <= max_length.
    Function returns None if the orbit is not of size cluster_size
        (aka this filter cannot be applied)


    Parameters
    ----------
    cluster_size: int
        Size (number of points) of clusters to apply filter to.
    max_length : float
        Maximum allowed length for two points within cluster.

    Returns
    -------
    sub_max_length_filter : function
        Function takes 1 argument: a dictionary of orbit data.
        The function will return a bool (or None) based on the input criteria.
    """

    def sub_max_length_filter(orbit_data: dict) -> bool:
        """Sub function for max_length_filter.
        Returns False if the orbit in `orbit_data` has `cluster_size` points
            and a maximum distance between points greater than `maximum_length`
        Returns True if the orbit in `orbit_data` has `cluster_size` points
            and a maximum distance between points less than `maximum_length`
        Returns None if the orbit is not of size cluster_size
            (aka this filter cannot be applied)

        Parameters
        ----------
        orbit_data : dict
            CASM basis function data for a specific orbit.

        Returns
        -------
        bool
            Whether the orbit passes the maximum length filter.
        """
        if len(orbit_data["prototype"]["sites"]) != cluster_size:
            return None
        else:
            return orbit_data["prototype"]["max_length"] <= max_length

    return sub_max_length_filter


def min_length_filter(cluster_size: int, min_length: float) -> Callable:
    """Returns function that takes orbit_data as input for the selection filtering.
    Returned function will return False if orbit of size cluster_size
        has minimum cluster size < min_length.
    Returned function will True if orbit of size cluster_size
        has minimum cluster size > min_length.
    Returned function will None if orbit is not of size cluster_size
        (aka this filter cannot be applied).

    Parameters
    ----------
    cluster_size: int
        Size (number of points) of clusters to apply filter to.
    min_length : float
        Minimum allowed length for two points within cluster.

    Returns
    -------
    sub_min_length_filter : function
        Function takes 1 argument: a dictionary of orbit data.
        The function will return a bool (or None) based on the input criteria.
    """

    def sub_min_length_filter(orbit_data: dict) -> bool:
        """Sub function for max_length_filter.
        Returns False if the orbit in `orbit_data` has `cluster_size` points
            and a minimum distance between points less than `min_length`.
        Returns True if the orbit in `orbit_data` has `cluster_size` points
            and a minimum distance between points greater than `min_length`.
        Returns None if the orbit is not of size cluster_size
            (aka this filter cannot be applied).

        Parameters
        ----------
        orbit_data : dict
            CASM basis function data for a specific orbit.

        Returns
        -------
        bool
            Whether the orbit passes the minimum length filter.
        """
        if len(orbit_data["prototype"]["sites"]) != cluster_size:
            return None
        else:
            return orbit_data["prototype"]["min_length"] >= min_length

    return sub_min_length_filter
