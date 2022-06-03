import os
import json
import numpy as np
import vtk
import pyvista
import meshio
import mpi4py
try:
    import dolfinx 
except ImportError:
    continue

def get_mesh_file_names_and_dirs(parent_dir):
    mesh_files = os.listdir(parent_dir)
    mesh_files = [file_ for file_ in mesh_files if '.json' in file_]
    mesh_dirs = [os.path.join(parent_dir, name) for name in mesh_files]
    mesh_names = [f'name_{name.split(".")[0]}' for name in mesh_files]
    return mesh_names, mesh_dirs

def load_meshes(mesh_dir):
    with open(mesh_dir, 'r') as f:
        meshes = json.load(f)
    input_mesh, output_mesh = meshes['input'], meshes['output']
    return input_mesh, output_mesh

def append_zero_col(x, col_idx):
    x = np.array(x)
    zeros_col = np.zeros((x.shape[0], 1))
    new_x = np.append(x, zeros_col, axis=1)
    new_x[:, [-1, col_idx]] = new_x[:, [col_idx, -1]]
    return new_x

def create_mesh(points, cells, cell_geom, mesh_type):
    if (mesh_type=='dict') or (mesh_type is None):
        mesh = {'points': points, 'cells': cells}
    elif mesh_type=='meshio':
        mesh = meshio.Mesh(points=points, cells={cell_geom: cells})
    elif mesh_type=='pyvista':
        # Pyvista throws error if mesh is 2d:
        if points.shape[-1] < 3:
            points = append_zero_col(points, col_idx=2)
        mesh = pyvista.UnstructuredGrid({vtk_idx[cell_geom]: cells}, points)
    elif mesh_type=='dolfinx':
        cells, domain = rearrange_cells_to_dolfinx_order(cells, points, cell_geom, return_domain=True)
        mesh = dolfinx.mesh.create_mesh(mpi4py.MPI.COMM_WORLD, cells, points, domain)
    else:
        raise ValueError('Invalid mesh type specified.')
    return mesh

vtk_idx = {'hexahedron': vtk.VTK_HEXAHEDRON, 'tetra': vtk.VTK_TETRA, 'quad': vtk.VTK_QUAD, 'triangle': vtk.VTK_TRIANGLE}
def unpack_mesh(mesh, cell_geom, mesh_type):
    if (mesh_type=='dict') or (mesh_type is None):
        points, cells = np.array(mesh['points']), np.array(mesh['cells'])
    elif mesh_type=='meshio':
        points, cells = mesh.points, mesh.cells_dict[cell_geom]
    elif mesh_type=='pyvista':
        points, cells = mesh.points, mesh.cells_dict[vtk_idx[cell_geom]]
    elif mesh_type=='dolfinx':
        import dolfinx
        V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
        points, cells = grid.points, grid.cells_dict[vtk_idx[cell_geom]]
    else:
        raise ValueError('Invalid input_mesh_type to be tested.')
    return points, cells

def create_tetridiv_inputs(mesh, input_mesh_type, output_mesh_type):
    points = np.array(mesh['points'])
    cells = np.array(mesh['cells'])
    cell_geom = mesh['cell_geom']
    if input_mesh_type is None:
        inputs = {'points': points, 'cells': cells}
    else:
        inputs = {'mesh': create_mesh(points, cells, cell_geom, input_mesh_type)}
    inputs['output_type'] = output_mesh_type
    return inputs

# See http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
gmsh_idx = {'hexahedron': 5, 'tetra': 4, 'quad': 3, 'triangle': 2}
def rearrange_cells_to_dolfinx_order(cells, points, cell_geom, return_domain=False):
    num_spatial_dim = points.shape[-1]
    domain = dolfinx.io.ufl_mesh_from_gmsh(gmsh_idx[cell_geom], num_spatial_dim)
    ufl_cell_type = dolfinx.cpp.mesh.to_type(str(domain.ufl_cell()))
    cells = cells[:, dolfinx.cpp.io.perm_gmsh(ufl_cell_type, cells.shape[1])]
    return cells, domain if return_domain else cells