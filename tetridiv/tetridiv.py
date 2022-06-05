"""Module containing `tet2hex` and `tri2quad`; both functions can be directly imported from `tetridiv` package, however."""

import importlib
import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Mesh = Any
_SUPPORTED_MESH_TYPES = ("dict", "meshio", "pyvista", "dolfinx")


def tet2hex(
    mesh: Optional[Mesh] = None,
    points: Optional[np.ndarray] = None,
    cells: Optional[np.ndarray] = None,
    output_type: Optional[str] = None,
) -> Mesh:
    r"""
    Subdivides tetrahedral meshes into hexahedral meshes.

    Either a `mesh` object of a supported type (i.e. `dict`, `meshio`, `pyvista`, or `dolfinx`) or both the `points` and `cells` arrays of the mesh must be specified. The type of the mesh object to be returned can be explicitly specified with `output_type`; if not provided, the type of the returned mesh is inferred. More specifically, if a supported `mesh` object is provided, the returned mesh will be of the same type; conversely, if a `points` array and `cells` array are provided, a dictionary with the subdivided mesh's `points` and `cells` is returned.

    Parameters
    ----------
    mesh : Mesh, optional
        Tetrahedral mesh object to be subdivided. Default: None
    points : (N,3) array_like, optional
        Vertex coordinates array of tetrahedral mesh to be subdivided, where N is the number of vertices in the mesh. `points[i]` should be the coordinates of the `i`th vertex. Default: None
    cells : (M,4) array_like, optional
        Vertex connectivity array of mesh to be subdivided, where M is the number of tetrahedral elements in the mesh. `cells[i]` should be the vertex indices which comprise the `i`th tetrahedron. Default: None
    output_type : str, optional
        Type of mesh object to return. If not specified, type of returned mesh is inferred. Default: None

    Return
    -------
    hexahedral_mesh : Mesh
        Subdivided mesh consisting of hexahedral elements. Type of returned mesh object is either inferred from input mesh or specified by `output_type`.

    Raises
    ------
    ValueError
        If a 'faulty' or 'incorrect' input mesh has been provided (e.g. non-tetrahedral mesh is provided).
    TypeError
        If provided `mesh` object is of an unsupported type, or if an unsupported `output_type` is requested.
    ImportError
        If a module required to output a requested mesh type has not been installed.

    Examples
    --------
    >>> import numpy as np
    >>> from tetridiv import tet2hex
    >>> points = np.array([[0., 0., 0.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])
    >>> cells = np.array([[0, 1, 2, 3]])
    >>> tet2hex(mesh={'points': points, 'cells': cells})
    {'cells': array([[ 0,  4, 10,  5,  6, 11, 14, 12],
                    [ 1,  7, 10,  4,  8, 13, 14, 11],
                    [ 2,  5, 10,  7,  9, 12, 14, 13],
                    [ 6, 11, 14, 12,  3,  8, 13,  9]]),
    'points': array([[0.        , 0.        , 0.        ],
                    [1.        , 0.        , 0.        ],
                    [0.        , 1.        , 0.        ],
                    [0.        , 0.        , 1.        ],
                    [0.5       , 0.        , 0.        ],
                    [0.        , 0.5       , 0.        ],
                    [0.        , 0.        , 0.5       ],
                    [0.5       , 0.5       , 0.        ],
                    [0.5       , 0.        , 0.5       ],
                    [0.        , 0.5       , 0.5       ],
                    [0.33333333, 0.33333333, 0.        ],
                    [0.33333333, 0.        , 0.33333333],
                    [0.        , 0.33333333, 0.33333333],
                    [0.33333333, 0.33333333, 0.33333333],
                    [0.25      , 0.25      , 0.25      ]])}
    >>> tet2hex(points=points, cells=cells)
    {'cells': array([[ 0,  4, 10,  5,  6, 11, 14, 12],
                    [ 1,  7, 10,  4,  8, 13, 14, 11],
                    [ 2,  5, 10,  7,  9, 12, 14, 13],
                    [ 6, 11, 14, 12,  3,  8, 13,  9]]),
    'points': array([[0.        , 0.        , 0.        ],
                    [1.        , 0.        , 0.        ],
                    [0.        , 1.        , 0.        ],
                    [0.        , 0.        , 1.        ],
                    [0.5       , 0.        , 0.        ],
                    [0.        , 0.5       , 0.        ],
                    [0.        , 0.        , 0.5       ],
                    [0.5       , 0.5       , 0.        ],
                    [0.5       , 0.        , 0.5       ],
                    [0.        , 0.5       , 0.5       ],
                    [0.33333333, 0.33333333, 0.        ],
                    [0.33333333, 0.        , 0.33333333],
                    [0.        , 0.33333333, 0.33333333],
                    [0.33333333, 0.33333333, 0.33333333],
                    [0.25      , 0.25      , 0.25      ]])}

    """
    hexahedral_mesh = _subdivide_mesh(mesh, points, cells, output_type, elem_type="tet")
    return hexahedral_mesh


def tri2quad(
    mesh: Optional[Mesh] = None,
    points: Optional[np.ndarray] = None,
    cells: Optional[np.ndarray] = None,
    output_type: Optional[str] = None,
) -> Mesh:
    r"""
    Subdivides triangular meshes into quarilateral meshes.

    Either a `mesh` object of a supported type (i.e. `dict`, `meshio`, `pyvista`, or `dolfinx`) or both the `points` and `cells` arrays of the mesh must be specified. The type of the mesh object to be returned can be explicitly specified with `output_type`; if not provided, the type of the returned mesh is inferred. More specifically, if a supported `mesh` object is provided, the returned mesh will be of the same type; conversely, if a `points` array and `cells` array are provided, a dictionary with the subdivided mesh's `points` and `cells` is returned.

    Parameters
    ----------
    mesh : Mesh, optional
        Triangular mesh object to be subdivided. Default: None
    points : (N,d) array_like, optional
        Vertex coordinates array of mesh to be subdivided, where N is the number of vertices in the mesh and `d` is the spatial dimensionality of the mesh (i.e. `d` = 2 or `d` = 3). `points[i]` should be the coordinates of the `i`th vertex. Default: None
    cells : (M,3) array_like, optional
        Vertex connectivity array of mesh to be subdivided, where M is the number of triangular elements in the mesh. `cells[i]` should be the vertex indices which comprise the `i`th triangle. Default: None
    output_type : str, optional
        Type of mesh object to return. If not specified, type of returned mesh is inferred. Default: None

    Return
    -------
    hexahedral_mesh : Mesh
        Subdivided mesh consisting of quadrilateral elements. Type of returned mesh object is either inferred from input mesh or specified by `output_type`.

    Raises
    ------
    ValueError
        If a 'faulty' or 'incorrect' input mesh has been provided (e.g. non-triangular mesh is provided).
    TypeError
        If provided `mesh` object is of an unsupported type, or if an unsupported `output_type` is requested.
    ImportError
        If a module required to output a requested mesh type has not been installed.

    Examples
    --------
    >>> import numpy as np
    >>> from tetridiv import tri2quad
    >>> points = np.array([[0., 0.],
                           [1., 0.],
                           [0., 1.]])
    >>> cells = np.array([[0, 1, 2]])
    >>> tri2quad(mesh={'points': points, 'cells': cells})
    {'cells': array([[0, 3, 6, 4],
                     [1, 5, 6, 3],
                     [6, 5, 2, 4]]),
     'points': array([[0.        , 0.        ],
                      [1.        , 0.        ],
                      [0.        , 1.        ],
                      [0.5       , 0.        ],
                      [0.        , 0.5       ],
                      [0.5       , 0.5       ],
                      [0.33333333, 0.33333333]])}
    >>> tri2quad(points=points, cells=cells)
    {'cells': array([[0, 3, 6, 4],
                     [1, 5, 6, 3],
                     [6, 5, 2, 4]]),
     'points': array([[0.        , 0.        ],
                      [1.        , 0.        ],
                      [0.        , 1.        ],
                      [0.5       , 0.        ],
                      [0.        , 0.5       ],
                      [0.5       , 0.5       ],
                      [0.33333333, 0.33333333]])}

    """
    quadrilateral_mesh = _subdivide_mesh(
        mesh, points, cells, output_type, elem_type="tri"
    )
    return quadrilateral_mesh


def _subdivide_mesh(
    input_mesh: Optional[Mesh],
    points: Optional[np.ndarray],
    cells: Optional[np.ndarray],
    output_type: Optional[str],
    elem_type: str,
) -> Mesh:
    r"""Subdivides tetrihedral mesh into hexahedral mesh (if `elem_type == 'tet'`) or a triangular mesh into a quadrilateral mesh (if `elem_type == 'tri'`)."""
    _verify_inputs(input_mesh, points, cells)
    input_mesh = _create_mesh_input(input_mesh, points, cells)
    input_mesh_type = _identify_mesh_type(input_mesh)
    output_type = _deduce_output_type(output_type, input_mesh_type)
    points, cells = _get_points_and_cells(input_mesh, input_mesh_type, elem_type)
    subdiv_points, subdiv_cells = _subdivide_points_and_cells(points, cells, elem_type)
    subdivided_mesh = _create_mesh(
        subdiv_points, subdiv_cells, input_mesh, output_type, elem_type
    )
    return subdivided_mesh


def _verify_inputs(
    mesh: Mesh, points: Optional[np.ndarray], cells: Optional[np.ndarray]
) -> None:
    r"""Ensure user specified either `mesh` OR `points` and `cells`."""
    if (mesh is None) and ((points is None) or (cells is None)):
        raise ValueError("Must specify either mesh OR points and cells as input(s).")
    elif all(val is not None for val in [mesh, points, cells]):
        raise ValueError("Must specify either mesh OR points and cells, but not both.")


def _create_mesh_input(
    mesh: Mesh, points: Optional[np.ndarray], cells: Optional[np.ndarray]
) -> Mesh:
    r"""Create mesh object from points and cells arrays if `mesh` is none."""
    if mesh is None:
        mesh = {"points": points, "cells": cells}
    if "tetgen" in str(type(mesh)):
        mesh = mesh.grid
    return mesh


def _identify_mesh_type(mesh: Mesh) -> str:
    r"""Return input mesh type as string."""
    class_name = str(type(mesh))
    input_type_matches = [name in class_name for name in _SUPPORTED_MESH_TYPES]
    if not any(input_type_matches):
        raise TypeError(
            f"""{class_name} meshes are not natively supported; instead, either explicitly specify the vertex coordinates (i.e. points) and topology (i.e. cells) of your mesh OR convert your mesh to one of the following formats: {', '.join(_SUPPORTED_MESH_TYPES)}."""
        )
    input_type = _take_first_match(input_type_matches)
    return input_type


def _deduce_output_type(output_type: Optional[str], input_type: str) -> str:
    r"""Return output mesh type as string."""
    if output_type is None:
        output_type = input_type
    else:
        output_type_name = str(output_type).lower()
        output_type_matches = [
            name in output_type_name for name in _SUPPORTED_MESH_TYPES
        ]
        if not any(output_type_matches):
            raise TypeError(
                f"'{output_type}' is not a valid output type; either leave output_type argument unspecified "
                f"or choose one of the following valid options: {', '.join(_SUPPORTED_MESH_TYPES)}."
            )
        output_type = _take_first_match(output_type_matches)
    return output_type


def _take_first_match(mesh_matches: List[bool]) -> str:
    r"""Get string."""
    mesh_type_idx = np.argmax(mesh_matches)
    mesh_type = _SUPPORTED_MESH_TYPES[mesh_type_idx]
    return mesh_type


#
#   Point & Cells Retrieval Methods
#


def _get_points_and_cells(
    mesh: Mesh, mesh_type: str, elem_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get points and cells arrays of `mesh`."""
    points = _get_points(mesh, mesh_type)
    cells = _get_cells(mesh, mesh_type, elem_type)
    cells = _ensure_cell_idxs_are_ints(cells)
    _check_points_and_cells(points, cells, elem_type)
    return points, cells


def _get_points(mesh: Mesh, mesh_type: str) -> np.ndarray:
    r"""Get points array of `mesh`."""
    if mesh_type == "dict":
        points = np.array(mesh["points"])
    elif mesh_type == "meshio":
        points = mesh.points
    elif mesh_type == "pyvista":
        points = np.array(mesh.points)
    elif mesh_type == "dolfinx":
        points = mesh.geometry.x
    return points


def _get_cells(mesh: Mesh, mesh_type: str, elem_type: str) -> np.ndarray:
    r"""Get cell array of `mesh`."""
    if mesh_type == "dict":
        cells = _get_cells_from_dict_mesh(mesh)
    elif mesh_type == "meshio":
        cell_key = "tetra" if elem_type == "tet" else "triangle"
        cells = _get_cells_from_cells_dict(
            mesh.cells_dict, cell_key, mesh_type, elem_type
        )
    elif mesh_type == "pyvista":
        import vtk

        cell_key = vtk.VTK_TETRA if elem_type == "tet" else vtk.VTK_TRIANGLE
        cells = _get_cells_from_cells_dict(
            mesh.cells_dict, cell_key, mesh_type, elem_type
        )
    elif mesh_type == "dolfinx":
        cells = _get_cells_from_dolfinx_mesh(mesh, elem_type)
    return cells


def _get_cells_from_dict_mesh(mesh_dict: Dict[str, np.ndarray]) -> np.ndarray:
    r"""Get cell array from a dict-type `mesh`."""
    try:
        cells = np.array(mesh_dict["cells"])
    except KeyError:
        raise ValueError("Mesh dictionary doesn't contain a 'cells' key.")
    return cells


def _get_cells_from_cells_dict(
    cells_dict: Dict[str, np.ndarray], key: str, mesh_type: str, elem_type: str
) -> np.ndarray:
    r"""Get cell array from cell dictionary of `mesh`."""
    try:
        cells = cells_dict[key]
    except KeyError:
        elem_name = "tetrahedral" if elem_type == "tet" else "triangular"
        raise ValueError(
            f"""cells_dict of {mesh_type} mesh does not contain {key} key; are you sure you provided a {elem_name} mesh?"""
        )
    return cells


def _get_cells_from_dolfinx_mesh(mesh: Mesh, elem_type: str) -> np.ndarray:
    r"""Get cell array from dolfinx-typed `mesh`."""
    # Could get segfault if we don't check that mesh is correct topology:
    actual_elem_type = str(mesh.topology.cell_name())
    if elem_type not in actual_elem_type:
        elem_name = "tetrahedral" if elem_type == "tet" else "triangular"
        raise ValueError(
            f"""dolfinx mesh consists of {actual_elem_type} cells, not {elem_name} cells. Are you sure you provided a {elem_name} mesh?"""
        )
    flattened_cells = mesh.geometry.dofmap.array
    vert_per_elem = 4 if elem_type == "tet" else 3
    cells = flattened_cells.reshape(-1, vert_per_elem)
    return cells


def _ensure_cell_idxs_are_ints(cells: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    r"""Convert `cells` array to `int`-typed array."""
    if np.any(np.mod(cells, 1) > epsilon):
        warnings.warn("cells contains floats; these values will be cast to integers.")
    return np.array(cells, dtype=int)


def _check_points_and_cells(
    coords: np.ndarray, cells: np.ndarray, elem_type: str
) -> None:
    r"""Throws error if `points` and `cells` are of an unexpected shape or contain unexpected values."""
    if coords.ndim != 2:
        raise ValueError(
            f"""Expected points to be a 2d array; instead, it was a {coords.ndim}d array."""
        )
    if (elem_type == "tet") & (coords.shape[-1] != 3):
        raise ValueError(
            f"""Expected points to be an N×3 array, since tetrahedral meshes exist in 3d space;
            instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a tetrahedral mesh?"""
        )
    if (elem_type == "tri") & (cells.shape[-1] != 3):
        raise ValueError(
            f"""Expected cells to be an N×3 array, since triangular elements consist of three vertices;
            instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a triangular mesh?"""
        )
    elif (elem_type == "tet") & (cells.shape[-1] != 4):
        raise ValueError(
            f"""Expected cells to be an N×4 array, since tetrahedral elements consist of four vertices;
            instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a tetrahedral mesh?"""
        )
    if np.min(cells) < 0:
        raise ValueError(
            f"""Expected smallest index in cells to be greater than or equal to 0; instead,
            the smallest value in cells was {np.min(cells)}."""
        )
    elif np.max(cells) > coords.shape[0] - 1:
        raise ValueError(
            f"""Expected largest index in cells to be an int less than or equal to (N-1), where N
            is the number of vertices listed in points; instead, the largest value in cells was {np.max(cells)},
            but only N = {coords.shape[0]} vertices were listed in points."""
        )


#
#   Subdivision Methods
#


def _subdivide_points_and_cells(
    points: np.ndarray, cells: np.ndarray, elem_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Subdivides points and cells arrays of tetrahedral meshes and triangular meshes into hexahedral and quadrilateral meshes respectively.

    The points and cells arrays are subdivided using a five-step process:
        1. Listing all of the unique element edges, faces, and volumes in the mesh (i.e. edges shared between multiple elements are only listed once). If a triangular mesh is supplied, only edges and faces are listed.
        2. Computing the centre points of all these unique edges, faces, and volumes; these centre points, along with all the vertices in the original mesh, are vertices in the subdivided mesh.
        3. Assign an index number to these new vertices.
        4. Group the new vertex indices by the the 'feature' (i.e. edge, face, volume) they were produced from. For example, all of the new vertex indices produced by finding the centrepoint of element 22's faces should be grouped together.
        5. Using the indice groupings from Step 4, assemble the connectivity matrix of the subdivided mesh.

    Parameters
    ----------
    points : (N,d) array_like
        Coordinates of mesh vertices, where N is the number of vertices in the mesh and d is the spatial dimensionality of the mesh (i.e. d = 3 if tetrahedral, and d = 2 or d = 3 if triangular).
    cells : (M,v) array_like
        Connectivity array of mesh, M is the number of elements and v is the number of vertices in each element (i.e. v = 4 if tetrahedral, and v = 3 if triangular).
    elem_type : {'tet', 'tri'}
        Element type of provided mesh: `tet` for tetrahedral, and `tri` for triangular.

    Return
    ------
    subdivided_points : (N*,d) array_like
        Coordinates of vertices in subdivided mesh, where N* is the number of vertices in the subdivided mesh.
    subdivided_cells : (s*N,d) array_like
        Connectivity array of subdivided mesh, where s is the number of new elements produced by subdividing a single element (i.e. s = 4 if subividing tetrahedral mesh, and s = 3 if subdividing triangular mesh).
    """
    # Get all possible combinations of local vertices which can be used to construct edges, faces, and volumes
    local_vert_combos = _create_vert_combos(elem_type)

    # Ensure tets correctly orientated so that local vertex indices are correct (i.e. 'highest' point labelled as vertex 3):
    if elem_type == "tet":
        cells = _order_verts_in_tet_by_height(points, cells, local_vert_combos)

    # Get global indices of vertices for each vertex in each edges, faces, and volumes:
    features_global_idxs = {
        feature_name: cells[:, vert_combo]
        for feature_name, vert_combo in local_vert_combos.items()
    }

    # Iterate over features to create new vertices at centre of features (i.e. mid-edge, mid-face, or mid-vol):
    N = points.shape[0]
    num_new_verts = []
    new_verts = {}
    new_verts_coords = []
    for feature, feat_pts in features_global_idxs.items():
        # feature: {'edge', 'face', 'vol'}
        # feat_pts.shape = (num_elems, num_features_per_elem, v)

        # Step 1:
        # List all features in mesh, including repeats:
        v = feat_pts.shape[-1]
        reshaped_feat_pts = feat_pts.reshape(-1, v)
        # shape = (num_features_per_elem*M, v)
        # Sort vertex numbers within each element:
        sorted_feat_pts = np.sort(reshaped_feat_pts, axis=-1)
        # Remove repeated features:
        unique_feat_pts, unique_inv = np.unique(
            sorted_feat_pts, axis=0, return_inverse=True
        )
        # shape = (num_unique_features, v)

        # Step 2:
        unique_feat_coords = points[unique_feat_pts, :]
        # shape = (num_unique_features, v, d)
        # Center of feature = average position of vertices in feature
        centre_pts = np.mean(unique_feat_coords, axis=1)
        new_verts_coords.append(centre_pts)  # shape = (num_unique_features, d)

        # Step 3:
        # Assign initial vertex indices starting from 0:
        new_vert_idx = np.arange(0, unique_feat_pts.shape[0])  # shape = (v,)
        # Shift values up so that they don't overlap with pre-existing vertex numbers:
        new_vert_idx = new_vert_idx + N + sum(num_new_verts)  # shape = (v,)
        num_new_verts.append(new_vert_idx.size)

        # Step 4:
        # 'Bring back' repeated vertices - see 'return_inverse' arg in np.unique documentation:
        repeated_new_verts = new_vert_idx[unique_inv]
        # shape = (M*num_features_per_elem,)
        # Associate each new vertex with the feature and element it was computed from (e.g. new_verts['face'][i,j] is the new index produced from the mid-point of the i'th elements j'th face)
        new_verts[feature] = repeated_new_verts.reshape(*feat_pts.shape[:-1])
        # shape = (M, num_features_per_elem)

    # Step 5:
    # New mesh coords = old mesh coords + coords of new vertices:
    subdivided_points = np.concatenate(
        [points, *new_verts_coords], axis=0
    )  # shape = (N+sum(num_new_vert), d)
    # Assemble cells array:
    if elem_type == "tet":
        # Global idxs of original verts in each tet = verts of tet volumes:
        orig_verts = features_global_idxs["vol"][:, 0, :]  # shape = (M, 4)
        subdivided_cells = _assemble_hexahedrons(
            orig_verts, new_verts, local_vert_combos
        )  # shape = (4M, 6)
    else:
        # Global idxs of original verts in each tri = verts of tri faces:
        orig_verts = features_global_idxs["face"][:, 0, :]  # shape = (M, 3)
        subdivided_cells = _assemble_quadrilaterals(
            orig_verts, new_verts, local_vert_combos
        )  # shape = (3M, 4)

    return subdivided_points, subdivided_cells


def _create_vert_combos(elem_type: str) -> Dict[str, List[Tuple[int, ...]]]:
    r"""Create dictionary of local vertex indice combinations for each feature (i.e. edge, face, volume) in a mesh element."""
    vert_per_cell = 4 if elem_type == "tet" else 3
    # Edge defined by 'picking' two vertices from element, face defined by 'picking' three vertices from element (e.g. if vert_per_cell = 3, then vert_combos['edge'] = [(0,1), (0,2), (1,2)]):
    vert_combos = {
        key: list(combinations(range(vert_per_cell), num_vert))
        for key, num_vert in {"edge": 2, "face": 3}.items()
    }
    if elem_type == "tet":
        # Tetrahedral volume defined by 4 vertices:
        vert_combos["vol"] = list(combinations(range(vert_per_cell), 4))
    return vert_combos


def _order_verts_in_tet_by_height(
    points: np.ndarray,
    cells: np.ndarray,
    local_vert_combos: Dict[str, List[Tuple[int, ...]]],
) -> np.ndarray:
    r"""Orders each row in cells array so that vertex indices appear in order of increasing z position."""
    # Get global indices of vertices which comprise each tet:
    vert_idx = cells[:, local_vert_combos["vol"]]
    # shape = (num_tet, num_vol_per_tet=1, num_vert_per_vol=4)
    # Get coordinates of these vertices:
    verts_coords = points[vert_idx, :]
    # shape = (num_tet, num_vol_per_tet=1, num_vert_per_vol=4, num_spatial_dim=3)
    # Sort vertices in each element by their z height:
    vert_z_pos = verts_coords[:, 0, :, -1]
    z_sort = np.argsort(vert_z_pos, axis=-1)
    cells = np.take_along_axis(cells, z_sort, axis=-1)
    return cells


def _assemble_hexahedrons(
    orig_verts: np.ndarray,
    new_verts: Dict[str, np.ndarray],
    local_vert_combos: Dict[str, List[Tuple[int, ...]]],
) -> np.ndarray:
    r"""
    Create cells array of subdivided hexahedron mesh.

    The cells array is created using a two-step process:
        1. For each local vertex indice, create an (M,)-shaped array (where M = number of tets) containing the global index of this local vertex in every tet. This means that we'd have 4 arrays corresponding to the original tet vertices, 6 arrays corresponding to the new mid-edge points, 4 corresponding to mid-face points, and 1 corresponding to the mid-volume point. As an example, the '012' mid-face array should contain the global index of the '012' mid-face point for every tet.
        2. The arrays created in Step 1 are a mapping from local vertex indices to global index indices; this allows us to build
        the subdivided mesh's hexahedrons by stacking these local-to-global index arrays.

    Parameters
    ----------
    orig_verts : (M,4) array_like
        Global indices of original vertices in every tet; for instance, orig_verts[20,2] should be the global index of the 2nd vertex in the 20th tet.
    new_verts : dict of array_like
        Dictionary containing global indices of mid-points of each feature (i.e. edge, face, volume) in every tet. For example, new_verts['face'][20,2] should be the global index of the 2nd mid-face point in the 20th tet.
    local_vert_combos : dict of lists of tuples
        Dictionary containing all local vertex indice combos corresponding to each feature.

    Return
    ------
    subdivided_cells : (4M,6) array_like
        Connectivity array of subdivided hexahedral mesh.
    """
    # Step 1:
    # Get global idx of original verts first:
    num_vert_in_vol = 4
    verts = {
        f"{vert_idx}": orig_verts[:, vert_idx] for vert_idx in range(num_vert_in_vol)
    }
    # verts dictionary is of the form:
    # {0: array with global idx of vert 0 in every tet, ...}
    # Get global idx of feature mid-points next:
    feature_dicts = []
    for feature in ("edge", "face", "vol"):
        vert_local_idx = local_vert_combos[feature]
        vert_idx_dict = {}
        # For each mid-point in
        for i, vert_idx_i in enumerate(vert_local_idx):
            # vert_idx_i is an array with vertex combos; concat this to a string:
            vert_idx_str = "".join(str(val) for val in vert_idx_i)
            vert_idx_dict[vert_idx_str] = new_verts[feature][:, i]  # shape = (M,)
        feature_dicts.append(vert_idx_dict)
    edges, faces, vols = feature_dicts
    # Dictionaries of the form:
    # edges = {'01': array with global idxs of '01' mid-edge point in every tet, ...}
    # faces = {'012': array with global idxs of '012' mid-face point in every tet, ...}
    # vols =  {'0123': array with global idxs of '0123' mid-vol point in every tet, ...}

    # Step 2:
    original_num_tet = orig_verts.shape[0]
    num_vert_per_hex = 8
    num_hex_per_tet = hpt = 4
    num_hex = num_hex_per_tet * original_num_tet
    subdivided_cells = np.zeros((num_hex, num_vert_per_hex))
    # Hexahedron 0 = Vert_0, Edge_01, Face_012, Edge_02, Edge_03, Face_013, Vol_0123, Face_023
    subdivided_cells[0::hpt, :] = np.stack(
        [
            verts["0"],
            edges["01"],
            faces["012"],
            edges["02"],
            edges["03"],
            faces["013"],
            vols["0123"],
            faces["023"],
        ],
        axis=-1,
    )
    # Hexahedron 1 = Vert_1, Edge_12, Face_012, Edge_01, Edge_13, Face_123, Vol_0123, Face_013
    subdivided_cells[1::hpt, :] = np.stack(
        [
            verts["1"],
            edges["12"],
            faces["012"],
            edges["01"],
            edges["13"],
            faces["123"],
            vols["0123"],
            faces["013"],
        ],
        axis=-1,
    )
    # Hexahedron 2 = Vert_2, Edge_02, Face_012, Edge_12, Edge_23, Face_023, Vol_0123, Face_123
    subdivided_cells[2::hpt, :] = np.stack(
        [
            verts["2"],
            edges["02"],
            faces["012"],
            edges["12"],
            edges["23"],
            faces["023"],
            vols["0123"],
            faces["123"],
        ],
        axis=-1,
    )
    # Hexahedron 3 = Edge_03, Face_013, Vol_0123, Face_023, Vert_3, Edge_13, Face_123, Edge_23
    subdivided_cells[3::hpt, :] = np.stack(
        [
            edges["03"],
            faces["013"],
            vols["0123"],
            faces["023"],
            verts["3"],
            edges["13"],
            faces["123"],
            edges["23"],
        ],
        axis=-1,
    )

    return subdivided_cells


def _assemble_quadrilaterals(
    orig_verts: np.ndarray,
    new_verts: Dict[str, np.ndarray],
    local_vert_combos: Dict[str, List[Tuple[int, ...]]],
) -> np.ndarray:
    r"""
    Create cells array of subdivided quadrilateral mesh.

    The cells array is created using a two-step process:
        1. For each local vertex indice, create an (M,)-shaped array (where M = number of tris) containing the global index of this local vertex in every tri. This means that we'd have 3 arrays corresponding to the original tri vertices, 3 arrays corresponding to the new mid-edge points, and 1 corresponding to mid-face point. As an example, the '01' mid-edge array should contain the global index of the '01' mid-edge point for every tri.
        2. The arrays created in Step 1 are a mapping from local vertex indices to global index indices; this allows us to build
        the subdivided mesh's quadrilaterals by stacking these local-to-global index arrays.

    Parameters
    ----------
    orig_verts : (M,3) array_like
        Global indices of original vertices in every tri; for instance, orig_verts[20,2] should be the global index of the 2nd vertex in the 20th tri.
    new_verts : dict of array_like
        Dictionary containing global indices of mid-points of each feature (i.e. edge and face) in every tri. For example, new_verts['edge'][20,2] should be the global index of the 2nd mid-face point in the 20th tri.
    local_vert_combos : dict of lists of tuples
        Dictionary containing all local vertex indice combos corresponding to each feature.

    Return
    ------
    subdivided_cells : (3M,4) array_like
        Connectivity array of subdivided quadrilateral mesh.
    """
    # Step 1:
    # Get global idx of original verts first:
    num_vert_in_face = 3
    verts = {
        f"{vert_idx}": orig_verts[:, vert_idx] for vert_idx in range(num_vert_in_face)
    }
    # verts dictionary is of the form:
    # {0: array with global idx of vert 0 in every tri, ...}
    feature_dicts = []
    for feature in ("edge", "face"):
        vert_local_idx = local_vert_combos[feature]
        vert_idx_dict = {}
        for i, vert_idx_i in enumerate(vert_local_idx):
            # vert_idx_i is an array with vertex combos; concat this to a string:
            vert_idx_str = "".join(str(val) for val in vert_idx_i)
            vert_idx_dict[vert_idx_str] = new_verts[feature][:, i]  # shape = (M,)
        feature_dicts.append(vert_idx_dict)
    edges, faces = feature_dicts
    # Dictionaries of the form:
    # edges = {'01': array with global idxs of '01' mid-edge point in every tri, ...}
    # faces = {'012': array with global idxs of '012' mid-face point in every tet, ...}

    # Step 2:
    original_num_tri = orig_verts.shape[0]
    num_vert_per_quad = 4
    num_quad_per_tri = qpt = 3
    num_quad = num_quad_per_tri * original_num_tri
    subdivided_cells = np.zeros((num_quad, num_vert_per_quad))
    # Quadrilateral 0 = Vert_0, Edge_01, Face_012, Edge_02
    subdivided_cells[0::qpt, :] = np.stack(
        [verts["0"], edges["01"], faces["012"], edges["02"]], axis=-1
    )
    # Quadrilateral 1 = Vert_1, Edge_12, Face_012, Edge_01
    subdivided_cells[1::qpt, :] = np.stack(
        [verts["1"], edges["12"], faces["012"], edges["01"]], axis=-1
    )
    # Quadrilateral 2 = Face_012, Edge_12, Vert_2, Edge_02
    subdivided_cells[2::qpt, :] = np.stack(
        [faces["012"], edges["12"], verts["2"], edges["02"]], axis=-1
    )

    return subdivided_cells


#
#   Post-Processing Methods
#


def _create_mesh(
    points: np.ndarray, cells: np.ndarray, mesh: Mesh, output_type: str, elem_type: str
) -> Mesh:
    r"""Create mesh of type `output_type` from provided `points` and `cells` arrays."""
    # Some mesh formats throw errors if cells doesn't consist of ints:
    cells = np.array(cells, dtype=int)

    if output_type == "dict":
        mesh = {"points": points, "cells": cells}

    elif output_type == "meshio":
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio must be installed to output a meshio mesh.")
        cell_key = "hexahedron" if elem_type == "tet" else "quad"
        mesh = meshio.Mesh(points=points, cells={cell_key: cells})

    elif output_type == "pyvista":
        try:
            import pyvista
            import vtk
        except ImportError:
            raise ImportError(
                "pyvista and vtk must be installed to output a pyvista mesh."
            )
        # Pyvista throws error if mesh is 2d:
        if points.shape[-1] < 3:
            points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)
        cell_key = vtk.VTK_HEXAHEDRON if elem_type == "tet" else vtk.VTK_QUAD
        mesh = pyvista.UnstructuredGrid({cell_key: cells}, points)

    elif output_type == "dolfinx":
        # Hexahedra and quadrilaterals are assigned unique identifying int - See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
        try:
            import dolfinx
            import mpi4py
        except ImportError:
            raise ImportError(
                "dolfinx and mpi4py must be installed to output a dolfinx mesh."
            )
        gmsh_mesh_num = 5 if elem_type == "tet" else 3
        num_spatial_dim = points.shape[-1]
        domain = dolfinx.io.ufl_mesh_from_gmsh(gmsh_mesh_num, num_spatial_dim)
        ufl_elem_type = dolfinx.cpp.mesh.to_type(str(domain.ufl_cell()))
        cells = cells[:, dolfinx.cpp.io.perm_gmsh(ufl_elem_type, cells.shape[1])]
        mesh = dolfinx.mesh.create_mesh(mpi4py.MPI.COMM_WORLD, cells, points, domain)

    return mesh
