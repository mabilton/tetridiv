from math import cos, pi, sin

import numpy as np
import pyvista

#
#   Rotation Methods
#


def rotate_vector(f, y_rot, x_rot=0):
    # Using Euler angles - see https://www.autonomousrobotslab.com/frame-rotations-and-representations.html
    rot_matrix = create_rot_matrix(y_rot, x_rot)
    rotated_f = rot_matrix @ f
    return rotated_f


def create_rot_matrix(y_rot, x_rot, ang_to_rad=pi / 180):
    # NB: Negative associated with y so increasing y_rot goes in 'right direction'
    theta, psi = -ang_to_rad * y_rot, ang_to_rad * x_rot
    rot_matrix = np.array(
        [
            [cos(theta), 0, -sin(theta)],
            [sin(psi) * sin(theta), cos(psi), sin(psi) * cos(theta)],
            [cos(psi) * sin(theta), -sin(psi), cos(psi) * cos(theta)],
        ]
    )
    return rot_matrix


#
#   Dolfinx Methods
#


def get_dolfinx_mesh_cells_at_query_points(query_points, mesh):
    import dolfinx

    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells, points_on_processors = [], []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, query_points)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, query_points
    )
    for i, point in enumerate(query_points):
        if len(colliding_cells.links(i)) > 0:
            points_on_processors.append(point)
            cells.append(colliding_cells.links(i)[0])
    return points_on_processors, cells


def create_polydata_from_pyvista_grid(grid, face_key):
    cells = grid.cells_dict[face_key]
    vert_per_face = cells.shape[-1]
    vert_per_face *= np.ones((cells.shape[0], 1), dtype=int)
    faces = np.concatenate([vert_per_face, cells], axis=1).flatten()
    polydata = pyvista.PolyData(grid.points, faces)
    return polydata
