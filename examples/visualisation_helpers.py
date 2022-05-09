import pyvista
import vtk
import numpy as np

def plot_meshio_mesh(mesh, cam_pos='xy', window_size=(960,480), sargs=None, clim=None, n_labels=5, label_decimals=0, zoom=1, slice_mesh=False):
    
    if sargs is None:
        sargs = _create_default_sargs(n_labels, label_decimals)
    
    grid = create_pyvista_grid_from_meshio(mesh)
    
    pyvista.start_xvfb()
    p = pyvista.Plotter(window_size=window_size)
    if slice_mesh:
        cell_centres = grid.cell_centers().points
        cells_to_remove = cell_centres[:,0] >= 0
        grid_rem = grid.remove_cells(cells_to_remove)
        grid_rem.clear_field_data()
        p.add_mesh(grid_rem, show_edges=True, color='white')
    else:
        actor_0 = p.add_mesh(grid, show_edges=True)
    p.show_axes()
    p.camera_position = cam_pos
    p.camera.roll += - 90
    p.camera.zoom(zoom)
    return p

def _create_default_sargs(n_labels, label_decimals):
    return dict(label_font_size=16, shadow=True, n_labels=n_labels,
                fmt=f"%.{label_decimals}f", font_family="arial")

_vtk_idx = {'hexahedron': vtk.VTK_HEXAHEDRON,
            'tetra': vtk.VTK_TETRA,
            'quad': vtk.VTK_QUAD,
            'triangle': vtk.VTK_TRIANGLE}
def create_pyvista_grid_from_meshio(mesh):
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
        raise ValueError('Unsupported mesh type.')
    return grid