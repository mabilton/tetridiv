import numpy as np
import pytest
import utils

import tetridiv

#
#   Set-Up
#

# Mesh types to test:
input_types_to_test = (None, "dict", "meshio", "pyvista")
output_types_to_test = (*input_types_to_test, "dolfinx")
# Methods to test:
tetridiv_methods = {"tet2hex": tetridiv.tet2hex, "tri2quad": tetridiv.tri2quad}
# List of mesh files:
tet2hex_test_names, tet2hex_test_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tet2hex"
)
tri2quad_test_3d_names, tri2quad_test_3d_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tri2quad/3d"
)
tri2quad_test_2d_names, tri2quad_test_2d_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tri2quad/2d"
)
tri2quad_test_dirs = [*tri2quad_test_3d_dirs, *tri2quad_test_2d_dirs]
tri2quad_test_names = [*tri2quad_test_3d_names, *tri2quad_test_2d_names]
all_tests_dirs = [*tet2hex_test_dirs, *tri2quad_test_dirs]
all_tests_names = [*tet2hex_test_names, *tri2quad_test_names]

#
#   Tests
#


@pytest.mark.parametrize(
    "mesh_dir",
    [*tri2quad_test_3d_dirs, *tri2quad_test_3d_dirs],
    ids=[*tri2quad_test_3d_names, *tri2quad_test_3d_names],
)
@pytest.mark.parametrize(
    "input_mesh_type",
    input_types_to_test,
    ids=[f"input_={type_}" for type_ in input_types_to_test],
)
@pytest.mark.parametrize(
    "req_output_mesh_type",
    output_types_to_test,
    ids=[f"output_={type_}" for type_ in output_types_to_test],
)
def test_tet2hex_nontetra_mesh(mesh_dir, input_mesh_type, req_output_mesh_type):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    with pytest.raises(ValueError, match="sure you provided a tetrahedral mesh?"):
        tetridiv_inputs = utils.create_tetridiv_inputs(
            input_mesh, input_mesh_type, req_output_mesh_type
        )
        tetridiv.tet2hex(**tetridiv_inputs)


@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
@pytest.mark.parametrize(
    "input_mesh_type",
    input_types_to_test,
    ids=[f"input_={type_}" for type_ in input_types_to_test],
)
@pytest.mark.parametrize(
    "req_output_mesh_type",
    output_types_to_test,
    ids=[f"output_={type_}" for type_ in output_types_to_test],
)
def test_tri2quad_nontri_mesh(mesh_dir, input_mesh_type, req_output_mesh_type):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    with pytest.raises(ValueError, match="sure you provided a triangular mesh?"):
        tetridiv_inputs = utils.create_tetridiv_inputs(
            input_mesh, input_mesh_type, req_output_mesh_type
        )
        tetridiv.tri2quad(**tetridiv_inputs)


invalid_meshes = {
    "int": 1,
    "array": np.ones((2, 2)),
    "str": "abc",
    "tuple": (1, "a"),
    "list": [2, "b"],
}


@pytest.mark.parametrize(
    "mesh",
    invalid_meshes.values(),
    ids=[f"mesh_={mesh}" for mesh in invalid_meshes.keys()],
)
@pytest.mark.parametrize(
    "tetridiv_method",
    tetridiv_methods.values(),
    ids=[f"method_={method}" for method in tetridiv_methods.keys()],
)
def test_unsupported_mesh_type(mesh, tetridiv_method):
    with pytest.raises(
        TypeError, match="convert your mesh to one of the following formats"
    ):
        tetridiv_method(mesh=mesh)


@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
def test_tet2hex_smallest_vertex_label_less_than_zero(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells -= np.min(cells) + 1
    with pytest.raises(ValueError, match="Expected smallest index in cells"):
        tetridiv.tet2hex(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", tri2quad_test_dirs, ids=tri2quad_test_names)
def test_tri2quad_smallest_vertex_label_less_than_zero(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells -= np.min(cells) + 1
    with pytest.raises(ValueError, match="Expected smallest index in cells"):
        tetridiv.tri2quad(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
def test_tet2hex_largest_vertex_label_too_large(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells += 1
    with pytest.raises(ValueError, match="Expected largest index in cells"):
        tetridiv.tet2hex(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", tri2quad_test_dirs, ids=tri2quad_test_names)
def test_tri2quad_largest_vertex_label_too_large(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells += 1
    with pytest.raises(ValueError, match="Expected largest index in cells"):
        tetridiv.tri2quad(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
def test_tet2hex_cells_not_ints(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells = np.array(cells, dtype=float) + 0.1
    with pytest.warns(UserWarning, match="cells contains floats"):
        tetridiv.tet2hex(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", tri2quad_test_dirs, ids=tri2quad_test_names)
def test_tri2quad_cells_not_ints(mesh_dir):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    points, cells = utils.unpack_mesh(
        input_mesh, input_mesh["cell_geom"], mesh_type="dict"
    )
    cells = np.array(cells, dtype=float) + 0.1
    with pytest.warns(UserWarning, match="cells contains floats"):
        tetridiv.tri2quad(points=points, cells=cells)


@pytest.mark.parametrize("mesh_dir", all_tests_dirs, ids=all_tests_names)
@pytest.mark.parametrize(
    "tetridiv_method",
    tetridiv_methods.values(),
    ids=[f"method_={method}" for method in tetridiv_methods.keys()],
)
def test_too_many_inputs(mesh_dir, tetridiv_method):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    with pytest.raises(
        ValueError, match="specify either mesh OR points and cells, but not both"
    ):
        tetridiv_method(
            mesh=input_mesh, points=input_mesh["points"], cells=input_mesh["cells"]
        )


@pytest.mark.parametrize(
    "tetridiv_method",
    tetridiv_methods.values(),
    ids=[f"method_={method}" for method in tetridiv_methods.keys()],
)
def test_no_inputs(tetridiv_method):
    with pytest.raises(
        ValueError, match="specify either mesh OR points and cells as input"
    ):
        tetridiv_method()


invalid_outputs = ("tetgen", "abc", 1, np.array([2]))


@pytest.mark.parametrize(
    "invalid_output_type",
    invalid_outputs,
    ids=[f"invalidoutput_{type_}" for type_ in invalid_outputs],
)
@pytest.mark.parametrize("mesh_dir", tri2quad_test_dirs, ids=tri2quad_test_names)
def test_invalid_output_mesh_type(mesh_dir, invalid_output_type):
    input_mesh, _ = utils.load_meshes(mesh_dir)
    with pytest.raises(TypeError, match="not a valid output type"):
        tetridiv.tri2quad(mesh=input_mesh, output_type=invalid_output_type)
