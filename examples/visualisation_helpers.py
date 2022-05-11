import pyvista
import vtk
import numpy as np
import utils

#
#   Main Methods
#

def plot_mesh(mesh, cam_pos='xy', window_size=(960,480), sargs=None, clim=None, n_labels=5, label_decimals=0, zoom=1, slice_mesh=False):
    
    if sargs is None:
        sargs = _create_default_sargs(n_labels, label_decimals)
    
    grid = create_pyvista_grid(mesh)
    
    pyvista.start_xvfb()
    p = pyvista.Plotter(window_size=window_size)
    if slice_mesh:
        cell_centres = grid.cell_centers().points
        mesh_centre = np.mean(cell_centres, axis=0)
        cells_to_remove = cell_centres[:,0] >= mesh_centre[0]
        grid_rem = grid.remove_cells(cells_to_remove)
        grid_rem.clear_field_data()
        p.add_mesh(grid_rem, show_edges=True, color='white')
    else:
        actor_0 = p.add_mesh(grid, show_edges=True)
    p.show_axes()
    p.camera_position = cam_pos
    p.camera.zoom(zoom)
    return p

def plot_deformation(u, mesh, rot_y=0, rot_x=0, cam_pos='xz', window_size=(960,480), sargs=None, clim=None, n_labels=5, 
label_decimals=0, zoom=1, deform_factor=1):
    
    if 'dolfinx' not in str(type(mesh)):
        raise ValueError('Must provide dolfinx mesh corresponding to solved displacements u.')

    if sargs is None:
        sargs = _create_default_sargs(n_labels, label_decimals)
    
    grid = create_pyvista_grid(mesh)
    
    # # Rotate grid so that gravity is pointing 'down':
    # grid = _rotate_pyvista_mesh(mesh, rot_y, rot_x)

    # Interpolate deformations at nodes:
    points_on_processors, cells = utils.get_dolfinx_mesh_cells_at_query_points(grid.points, mesh)
    u_def = u.eval(points_on_processors, cells)
    grid.point_data["Deformation / mm"] = u_def
    
    pyvista.start_xvfb()
    p = pyvista.Plotter(window_size=window_size)
    # Show undeformed mesh as wireframe:
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    # Plot deformed mesh:
    warped = grid.warp_by_vector("Deformation / mm", factor=deform_factor)
    actor_1 = p.add_mesh(warped, clim=clim, scalar_bar_args=sargs)
    p.show_axes()
    p.camera_position = cam_pos
    p.camera.zoom(zoom)
    return p

#
#   Mesh Conversion
#

def create_pyvista_grid(mesh):
    mesh_class = str(type(mesh))
    if 'dolfinx' in str(type(mesh)):
        grid = _create_grid_from_dolfinx(mesh)
    elif 'meshio' in str(type(mesh)):
        grid = _create_grid_from_meshio(mesh)
    elif 'pyvista' in str(type(mesh)):
        grid = mesh
    else:
        raise ValueError('Invalid mesh type; must choose between dolfinx, meshio, or pyvista mesh.')
    return grid

def _create_grid_from_dolfinx(mesh):
    import dolfinx 
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
    return grid

_vtk_idx = {'hexahedron': vtk.VTK_HEXAHEDRON,
            'tetra': vtk.VTK_TETRA,
            'quad': vtk.VTK_QUAD,
            'triangle': vtk.VTK_TRIANGLE}
def _create_grid_from_meshio(mesh):
    identified_mesh_type = False
    for cell_key, vtk_idx in _vtk_idx.items():
        if cell_key in mesh.cells_dict.keys():
            cells = np.array(mesh.cells_dict[cell_key], dtype=int)
            points = mesh.points
            if points.shape[-1] == 2:
                zeros = np.zeros((points.shape[0], 1))
                points = np.concatenate([points, zeros], axis=1)
            grid = pyvista.UnstructuredGrid({vtk_idx: cells}, points)
            identified_mesh_type = True
            break
    if not identified_mesh_type:
        raise ValueError('Unsupported meshio element type.')
    return grid

#
#   Misc
#

def _create_default_sargs(n_labels, label_decimals):
    return dict(label_font_size=16, shadow=True, n_labels=n_labels,
                fmt=f"%.{label_decimals}f", font_family="arial")

def _rotate_pyvista_mesh(mesh, rot_y, rot_x):
    cells_dict = mesh.cells_dict
    rotation_mat = utils.create_rot_matrix(rot_y, rot_x)
    points = rotation_mat @ mesh.points
    return pyvista.UnstructuredGrid(cells_dict, points)