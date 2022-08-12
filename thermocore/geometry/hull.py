import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from typing import List, Tuple, Sequence


def barycentric_coordinates(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Returns the barycentric coordinates of `point` with respect to the simplex defined by `vertices`.

    TODO: Take multiple points?

    Parameters
    ----------
    point : np.ndarray of floats, shape (n_dim,)
        Point to get barycentric coordinates for.
    vertices : np.ndarray of floats, shape (n_dim + 1, n_dim)
        Vertices of reference simplex for the barycentric coordinates.

    Returns
    -------
    np.ndarray of floats, shape (n_dim + 1,)
        Barycentric coordinates of `point`.
    """
    # Check dimensions
    n_dim = len(point)
    if not vertices.shape[0] == n_dim + 1:
        raise ValueError("Number of vertices provided inconsistent with dimension.")

    if not vertices.shape[1] == n_dim:
        raise ValueError("Point and vertex dimensions inconsistent.")

    # Find barycentric coordinates
    H = np.vstack((vertices.transpose(), np.ones((1, n_dim + 1))))
    H_inv = np.linalg.inv(H)
    return H_inv @ np.append(point, 1)


def inside_convex_hull(points: np.ndarray, test_points: np.ndarray) -> List[bool]:
    """Returns a list of booleans indicating whether each point in `test_points` is inside the convex hull of `points`.

    This does not require finding the convex hull of `points`, only determining whether each of `test_points`
    can be expressed as a convex combination of `points`, which can be done using linear programming.

    For a single test point p and the points x_i, we check whether there exist coefficients a_i such that
    p = a_1*x_1 + a_2*x_2 + ... + a_n*x_n, where the a_i are non-negative and sum to 1.

    Adapted from: https://stackoverflow.com/a/43564754

    Parameters
    ---------
    points : np.ndarray of floats, shape (n_points, n_dim)
        Points defining the convex hull.
    test_points : np.ndarray of floats, shape (n_test_points, n_dim)
        Points to be tested for whether they are inside the convex hull.

    Returns
    -------
    list[bool]
        Booleans indicating whether each test point is inside the convex hull.
    """
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.vstack((points.transpose(), np.ones((1, n_points))))
    return [linprog(c, A_eq=A, b_eq=np.append(p, 1.0)).success for p in test_points]


def full_hull(
    compositions: np.ndarray, energies: np.ndarray, qhull_options=None
) -> ConvexHull:
    """Returns the full convex hull of the points specified by appending `energies` to `compositions`.

    Parameters
    ----------
    compositions: np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points.
    energies: np.ndarray of floats, shape (n_points,)
        Energies of points.
    qhull_options: str
        Additional optionals that can be passed to Qhull. See details on the scipy.spatial.ConvexHull documentation. Default=None
    Returns
    -------
    ConvexHull
        Convex hull of points.
    """
    return ConvexHull(
        np.hstack((compositions, energies[:, np.newaxis])), qhull_options=qhull_options
    )


def lower_hull(
    convex_hull: ConvexHull, tolerance: float = 1e-14
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the vertices and simplices of the lower convex hull (with respect to the last coordinate) of `convex_hull`.

    Parameters
    ----------
    convex_hull : ConvexHull
        Complete convex hull object.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    lower_hull_vertex_indices : np.ndarray of ints, shape (n_vertices,)
        Indices of points forming the vertices of the lower convex hull.
    lower_hull_simplex_indices : np.ndarray of ints, shape (n_simplices,)
        Indices of simplices (within `convex_hull.simplices`) forming the facets of the lower convex hull.
    """
    # Find lower hull simplices
    lower_hull_simplex_indices = (-convex_hull.equations[:, -2] > tolerance).nonzero()[
        0
    ]
    if lower_hull_simplex_indices.size == 0:
        raise RuntimeError("No lower hull simplices found.")

    # Gather lower hull vertices from simplices
    lower_hull_vertex_indices = np.unique(
        np.ravel(convex_hull.simplices[lower_hull_simplex_indices])
    )
    return lower_hull_vertex_indices, lower_hull_simplex_indices


def simplex_energy_equation_matrix(
    convex_hull: ConvexHull, simplex_indices: Sequence[int], tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns a matrix that encodes the energy equation of each requested convex hull simplex.

    Each row in the matrix corresponds to a simplex (as specified by `simplex_indices`).

    Each simplex is described by a hyperplane. For example, in a 3d composition space
    a1*x1 + a2*x2 + a3*x3 + b*e + c = 0
    where x1, x2, x3 are the composition variables and e is the energy.

    Solving for e yields
    e = -(a1*x1 + a2*x2 + a3*x3 + c)/b

    The corresponding row in the equation matrix is then (-a1/b, -a2/b, -a3/b, -c/b),
    such that multiplying with the column vector (x1, x2, x3, 1) yields e.

    Parameters
    ----------
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    simplex_indices : Sequence[int]
        Indices of simplices (within `convex_hull.simplices`) to include in matrix.
    tolerance : float, optional
        Tolerance for checking for vertical hull facets with b close to 0 (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_simplex_indices, n_composition_axes + 1)
        Matrix of simplex energy equations.
    """
    # Check for vertical simplices (b = 0 case)
    vertical_simplex_indices = np.array(simplex_indices)[
        (np.abs(convex_hull.equations[simplex_indices, -2]) < tolerance).nonzero()[0]
    ]
    if not vertical_simplex_indices.size == 0:
        raise ValueError(
            f"Vertical hull simplex encountered: Simplex index {','.join(map(str, vertical_simplex_indices))}."
        )

    # Form and return equation matrix
    return (
        -np.delete(convex_hull.equations[simplex_indices, :], -2, axis=1)
        / convex_hull.equations[simplex_indices, -2][:, np.newaxis]
    )


def lower_hull_simplex_containing(
    compositions: np.ndarray,
    convex_hull: ConvexHull,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the lower convex hull simplices of `convex_hull` containing the points in composition space specified by `compositions`, and the corresponding energies.

    For points incident with multiple simplices, one of the simplices is chosen arbitrarily.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to find containing simplices for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    simplex_indices : np.ndarray of ints, shape (n_points,)
        Indices of simplices (within `convex_hull.simplices`) containing each point.
    energies : np.ndarray of floats, shape (n_points,)
        Energy values of points specified by `compositions` on their respective simplices.
    """
    if lower_hull_simplex_indices is None:
        lower_hull_simplex_indices = lower_hull(convex_hull, tolerance=tolerance)[1]

    # Promote 1D composition array to 2D array, if necessary
    if compositions.ndim == 1:
        compositions = compositions[:, np.newaxis]

    # Check that matrices are compatible with one another
    hull_composition_dimension = convex_hull.points.shape[1] - 1
    if not compositions.shape[1] == hull_composition_dimension:
        raise ValueError(
            f"Composition dimensions of input points and hull points differ: {compositions.shape[1]} vs {hull_composition_dimension}."
        )

    # Check composition bounds for input points
    # TODO: Should maybe use lower hull vertices rather than convex_hull.vertices in case of tolerance issues
    out_of_bounds = ~np.array(
        inside_convex_hull(convex_hull.points[convex_hull.vertices, :-1], compositions)
    )
    out_of_bounds_point_indices = out_of_bounds.nonzero()[0]
    if not out_of_bounds_point_indices.size == 0:
        raise ValueError(
            f"Point outside of hull composition bounds encountered: Point index {','.join(map(str, out_of_bounds_point_indices))}."
        )

    # Form equation matrix and multiply with compositions
    lower_hull_equation_matrix = simplex_energy_equation_matrix(
        convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )
    configuration_simplex_energies = lower_hull_equation_matrix @ np.vstack(
        [compositions.transpose(), np.ones(compositions.shape[0])]
    )

    # Identify and extract correct simplices, energies
    maximum_energy_simplex_indices = np.argmax(configuration_simplex_energies, axis=0)
    simplex_indices = lower_hull_simplex_indices[maximum_energy_simplex_indices]
    energies = np.take_along_axis(
        configuration_simplex_energies,
        maximum_energy_simplex_indices[np.newaxis, :],
        axis=0,
    )[0]
    return simplex_indices, energies


def lower_hull_energies(
    compositions: np.ndarray,
    convex_hull: ConvexHull,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns energies of points in composition space specified by `compositions` along the lower hull of `convex_hull`.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to get energies for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_points,)
        Energies of points.
    """
    return lower_hull_simplex_containing(
        compositions, convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )[1]


def lower_hull_distances(
    compositions: np.ndarray,
    energies: np.ndarray,
    convex_hull: ConvexHull = None,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns hull distances (energy above lower convex hull of `convex_hull`) of points in energy-composition space specified by `compositions` and `energies`.

    If `convex_hull` is omitted, it will be calculated from `compositions` and `energies`.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to get hull distances for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    energies : np.ndarray of floats, shape (n_points,)
        Energies of points to get hull distances for.
    convex_hull : ConvexHull, optional
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_points,)
        Hull distances of points.
    """
    if convex_hull is None:
        convex_hull = full_hull(compositions, energies)
        lower_hull_simplex_indices = None
    return energies - lower_hull_energies(
        compositions, convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )


def hull_distance_correlations(
    corr: np.ndarray,
    compositions: np.ndarray,
    formation_energy: np.ndarray,
    hull: ConvexHull = False,
) -> np.ndarray:
    """Calculated the effective correlations to predict hull distance instead of absolute formation energy.
    Parameters:
    -----------
    corr: np.array
        nxk correlation matrix, where n is the number of configurations and k is the number of ECI.
    comp: np.array
        nxc matrix of compositions, where n is the number of configurations and c is the number of composition axes.
    formation_energy: np.array
        nx1 matrix of formation energies.

    Returns:
    --------
    hulldist_corr: np.array
        nxk matrix of effective correlations describing hull distance instead of absolute formation energy. n is the number of configurations and k is the number of ECI.
    """

    # Build convex hull from compositions and formation energies
    if hull == False:
        hull = full_hull(compositions=compositions, energies=formation_energy)

    # Get convex hull simplices
    lower_vertices, lower_simplices = lower_hull(hull)

    hulldist_corr = np.zeros(corr.shape)

    for config_index in list(range(corr.shape[0])):

        # Find the simplex that contains the current configuration's composition, and find the hull energy for that composition
        relevant_simplex_index, hull_energy = lower_hull_simplex_containing(
            compositions=compositions[config_index].reshape(1, -1),
            convex_hull=hull,
            lower_hull_simplex_indices=lower_simplices,
        )

        relevant_simplex_index = relevant_simplex_index[0]

        # Find vectors defining the corners of the simplex which contains the curent configuration's composition.
        simplex_corners = compositions[hull.simplices[relevant_simplex_index]]
        interior_point = np.array(compositions[config_index]).reshape(1, -1)

        # Find barycentric coordinates of the interior point in composition space
        weights = barycentric_coordinates(
            point=interior_point, vertices=simplex_corners
        )

        # Form the hull distance correlations by taking a linear combination of simplex corners.
        hulldist_corr[config_index] = (
            corr[config_index] - weights @ corr[hull.simplices[relevant_simplex_index]]
        )

    return hulldist_corr
