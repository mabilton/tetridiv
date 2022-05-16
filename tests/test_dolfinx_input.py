import pytest
import dolfinx 
import mpi4py
import tetridiv
import utils
import numpy as np

tet2hex_test_names, tet2hex_test_dirs = utils.get_mesh_file_names_and_dirs('./tests/test_meshes/dolfinx/tet2hex')
tri2quad_3d_test_names, tri2quad_3d_test_dirs = utils.get_mesh_file_names_and_dirs('./tests/test_meshes/dolfinx/tri2quad/3d')
tri2quad_2d_test_names, tri2quad_2d_test_dirs = utils.get_mesh_file_names_and_dirs('./tests/test_meshes/dolfinx/tri2quad/2d')

@pytest.mark.parametrize("mesh_dir", tet2hex_test_dirs, ids=tet2hex_test_names)
def test_tet2hex_dolfinx_as_input(mesh_dir):
    input_mesh_type = req_output_mesh_type = 'dolfinx'
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    tetridiv_inputs = utils.create_tetridiv_inputs(input_mesh, input_mesh_type, req_output_mesh_type)
    output_mesh = tetridiv.tet2hex(**tetridiv_inputs)
    assert_meshes_are_same(output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type)

@pytest.mark.parametrize("mesh_dir", tri2quad_3d_test_dirs, ids=tri2quad_3d_test_names)
def test_tri2quad_3d_dolfinx_as_input(mesh_dir):
    input_mesh_type = req_output_mesh_type = 'dolfinx'
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    tetridiv_inputs = utils.create_tetridiv_inputs(input_mesh, input_mesh_type, req_output_mesh_type)
    output_mesh = tetridiv.tri2quad(**tetridiv_inputs)
    assert_meshes_are_same(output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type)

@pytest.mark.parametrize("mesh_dir", tri2quad_2d_test_dirs, ids=tri2quad_2d_test_names)
def test_tri2quad_2d_dolfinx_as_input(mesh_dir):
    input_mesh_type = req_output_mesh_type = 'dolfinx'
    input_mesh, correct_output_mesh = utils.load_meshes(mesh_dir)
    tetridiv_inputs = utils.create_tetridiv_inputs(input_mesh, input_mesh_type, req_output_mesh_type)
    output_mesh = tetridiv.tri2quad(**tetridiv_inputs)
    assert_meshes_are_same(output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type)

def assert_meshes_are_same(output_mesh, correct_output_mesh, input_mesh_type, req_output_mesh_type):
    # Unpack points and cells of meshes:
    output_type = input_mesh_type if req_output_mesh_type is None else req_output_mesh_type
    output_cell_geom = correct_output_mesh['cell_geom']
    output_points, output_cells = utils.unpack_mesh(output_mesh, output_cell_geom, mesh_type=output_type)
    correct_points, correct_cells = utils.unpack_mesh(correct_output_mesh, output_cell_geom, mesh_type='dict')
    # Ensure coordinates of vertices are equal:
    assert np.allclose(output_points, correct_points)
    # Ensure vertex indices are identical:
    assert np.all(output_cells == correct_cells)
    # Cells should contain integer values:
    assert 'int' in str(output_cells.dtype)