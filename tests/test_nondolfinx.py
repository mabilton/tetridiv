import numpy as np
import pytest
import utils

import tetridiv

#
#   Set-Up
#

input_types_to_test = (None, "dict", "meshio", "pyvista")
output_types_to_test = input_types_to_test
tet2hex_test_names, tet2hex_test_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tet2hex"
)
tri2quad_3d_test_names, tri2quad_3d_test_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tri2quad/3d"
)
tri2quad_2d_test_names, tri2quad_2d_test_dirs = utils.get_mesh_file_names_and_dirs(
    "./tests/test_meshes/nondolfinx/tri2quad/2d"
)

#
#   Tests
#


@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
@pytest.mark.parametrize(
    "input_mesh_type",
    input_types_to_test,
    ids=[f"inputmeshtype_{type_}" for type_ in input_types_to_test],
)
@pytest.mark.parametrize(
    "req_output_mesh_type",
    output_types_to_test,
    ids=[f"outputmeshtype_{type_}" for type_ in output_types_to_test],
)
def test_tet2hex_nondolfinx(mesh_dir, input_mesh_type, req_output_mesh_type):
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    tetridiv_inputs = utils.create_tetridiv_inputs(
        input_mesh, input_mesh_type, req_output_mesh_type
    )
    output_mesh = tetridiv.tet2hex(**tetridiv_inputs)
    assert_meshes_are_same(
        output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type
    )


@pytest.mark.parametrize("mesh_dir", tri2quad_3d_test_dirs, ids=tri2quad_3d_test_names)
@pytest.mark.parametrize(
    "input_mesh_type",
    input_types_to_test,
    ids=[f"inputmeshtype_{type_}" for type_ in input_types_to_test],
)
@pytest.mark.parametrize(
    "req_output_mesh_type",
    output_types_to_test,
    ids=[f"outputmeshtype_{type_}" for type_ in output_types_to_test],
)
def test_tri2quad_3d_nondolfinx(mesh_dir, input_mesh_type, req_output_mesh_type):
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    tetridiv_inputs = utils.create_tetridiv_inputs(
        input_mesh, input_mesh_type, req_output_mesh_type
    )
    output_mesh = tetridiv.tri2quad(**tetridiv_inputs)
    assert_meshes_are_same(
        output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type
    )


zero_col_idx = [None, *list(range(3))]


@pytest.mark.parametrize("mesh_dir", tri2quad_2d_test_dirs, ids=tri2quad_2d_test_names)
@pytest.mark.parametrize(
    "input_mesh_type",
    input_types_to_test,
    ids=[f"inputmeshtype_{type_}" for type_ in input_types_to_test],
)
@pytest.mark.parametrize(
    "req_output_mesh_type",
    output_types_to_test,
    ids=[f"outputmeshtype_{type_}" for type_ in output_types_to_test],
)
@pytest.mark.parametrize(
    "zero_col_idx", zero_col_idx, ids=[f"zerocolidx_{axis}" for axis in zero_col_idx]
)
def test_tri2quad_2d_nondolfinx(
    mesh_dir, input_mesh_type, req_output_mesh_type, zero_col_idx
):
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    if zero_col_idx is not None:
        # Add additional zero col to points so it's a 3d matrix:
        input_mesh["points"] = utils.append_zero_col(
            input_mesh["points"], col_idx=zero_col_idx
        )
        correct_output_mesh["points"] = utils.append_zero_col(
            correct_output_mesh["points"], col_idx=zero_col_idx
        )
    tetridiv_inputs = utils.create_tetridiv_inputs(
        input_mesh, input_mesh_type, req_output_mesh_type
    )
    output_mesh = tetridiv.tri2quad(**tetridiv_inputs)
    assert_meshes_are_same(
        output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type
    )


def assert_meshes_are_same(
    output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type
):
    # Unpack points and cells of meshes:
    output_type = (
        input_mesh_type if req_output_mesh_type is None else req_output_mesh_type
    )
    output_cell_geom = correct_output_mesh["cell_geom"]
    output_points, output_cells = utils.unpack_mesh(
        output_mesh, output_cell_geom, mesh_type=output_type
    )
    correct_points, correct_cells = process_correct_mesh(
        correct_output_mesh, output_cell_geom, input_mesh_type, output_type
    )
    # Ensure coordinates of vertices are equal:
    assert np.allclose(output_points, correct_points)
    # Ensure vertex indices are identical:
    assert np.all(output_cells == correct_cells)
    # Cells should contain integer values:
    assert "int" in str(output_cells.dtype)


def process_correct_mesh(correct_mesh, cell_geom, input_mesh_type, output_type):
    points, cells = utils.unpack_mesh(correct_mesh, cell_geom, mesh_type="dict")
    for mesh_type in [input_mesh_type, output_type]:
        mesh = utils.create_mesh(points, cells, cell_geom, mesh_type)
        points, cells = utils.unpack_mesh(mesh, cell_geom, mesh_type)
    return points, cells
