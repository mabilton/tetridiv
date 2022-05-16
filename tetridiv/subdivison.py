import warnings
import importlib
from itertools import combinations
import numpy as np

#
#   Main routines
#

def tet2hex(mesh=None, points=None, cells=None, output_type=None):
    return _main(mesh, points, cells, output_type, cell_geom='tet')
    
def tri2quad(mesh=None, points=None, cells=None, output_type=None):
    return _main(mesh, points, cells, output_type, cell_geom='tri')

def _main(mesh, points, cells, output_type, cell_geom):
    _verify_inputs(mesh, points, cells)
    mesh = _create_mesh_input(mesh, points, cells)
    input_mesh_type = _identify_mesh_type(mesh)
    output_type = _deduce_output_type(output_type, input_mesh_type)
    points, cells = _get_points_and_cells(mesh, input_mesh_type, cell_geom)
    points, cells = _subdivide_mesh(points, cells, cell_geom)
    return _create_mesh(points, cells, mesh, output_type, cell_geom)

def _verify_inputs(mesh, points, cells):
    if (mesh is None) and ((points is None) or (cells is None)):
        raise ValueError('Must specify either mesh OR points and cells as input(s).')
    elif all(val is not None for val in [mesh, points, cells]):
        raise ValueError('Must specify either mesh OR points and cells, but not both.')

def _create_mesh_input(mesh, points, cells):
    if mesh is None:
        mesh = {'points': points, 'cells': cells}
    if 'tetgen' in str(type(mesh)):
        mesh = mesh.grid
    return mesh

_supported_mesh_types = ('dict', 'meshio', 'pyvista', 'dolfinx')
def _identify_mesh_type(mesh):
    class_name = str(type(mesh))
    input_type_matches = [name in class_name for name in _supported_mesh_types]
    if not any(input_type_matches):
        raise ValueError(f'{class_name} meshes are not natively supported; instead, either explicitly specify ' 
                          "the vertex coordinates (i.e. points) and topology (i.e. cells) of your mesh OR "
                         f'convert your mesh to one of the following formats: {", ".join(_supported_mesh_types)}.')
    input_type = _take_first_match(input_type_matches)
    return input_type

def _deduce_output_type(output_type, input_type):
    if output_type is None:
        output_type = input_type
    else:
        output_type_name = str(output_type).lower()
        output_type_matches = [name in output_type_name for name in _supported_mesh_types]
        if not any(output_type_matches):
            raise ValueError(f"'{output_type}' is not a valid output type; either leave output_type argument unspecified "
                             f"or choose one of the following valid options: {', '.join(_supported_mesh_types)}.")
        output_type = _take_first_match(output_type_matches)
    return output_type

def _take_first_match(mesh_matches):
    mesh_type_idx = np.argmax(mesh_matches)
    mesh_type = _supported_mesh_types[mesh_type_idx]
    return mesh_type

#
#   Point & Cells Retrieval Methods
#

def _get_points_and_cells(mesh, mesh_type, cell_geom):
    points = _get_points(mesh, mesh_type)
    cells = _get_cells(mesh, mesh_type, cell_geom)
    cells = _ensure_cell_idxs_are_ints(cells)
    _check_points_and_cells(points, cells, cell_geom)
    return points, cells

def _get_points(mesh, mesh_type):
    if mesh_type=='dict':
        points = np.array(mesh['points'])
    elif mesh_type=='meshio':
        points = mesh.points
    elif mesh_type=='pyvista':
        points = np.array(mesh.points)
    elif mesh_type=='dolfinx':
        points = mesh.geometry.x
    return points
    
def _get_cells(mesh, mesh_type, cell_geom):
    if mesh_type=='dict':
        cells = _get_cells_from_mesh_dict(mesh)
    elif mesh_type=='meshio':
        cell_key = 'tetra' if cell_geom=='tet' else 'triangle'
        cells = _get_cells_from_cells_dict(mesh.cells_dict, cell_key, mesh_type, cell_geom)
    elif mesh_type=='pyvista':
        import vtk
        cell_key = vtk.VTK_TETRA if cell_geom=='tet' else vtk.VTK_TRIANGLE
        cells = _get_cells_from_cells_dict(mesh.cells_dict, cell_key, mesh_type, cell_geom)
    elif mesh_type=='dolfinx':
        cells = _get_cells_from_dolfinx_mesh(mesh, cell_geom)
    return cells    

def _get_cells_from_mesh_dict(mesh_dict):
    try:
        cells = np.array(mesh_dict['cells'])
    except KeyError:
        raise KeyError("Mesh dictionary doesn't contain a 'cells' key.")
    return cells

def _get_cells_from_cells_dict(cells_dict, key, mesh_type, cell_geom):
    try:
        cells = cells_dict[key]
    except KeyError:
        elem_name = 'tetrahedral' if cell_geom == 'tet' else 'triangular'   
        raise ValueError(f"cells_dict of {mesh_type} mesh does not contain {key} key; "
                         f"are you sure you provided a {elem_name} mesh?")
    return cells

def _get_cells_from_dolfinx_mesh(mesh, cell_geom):
    # Could get segfault if we don't check that mesh is correct topology:
    actual_cell_geom = str(mesh.topology.cell_name())
    if cell_geom not in actual_cell_geom:
        elem_name = 'tetrahedral' if cell_geom == 'tet' else 'triangular'  
        raise ValueError(f'dolfinx mesh consists of {actual_cell_geom} cells, not {elem_name} cells. ' 
                         f'Are you sure you provided a {elem_name} mesh?')
    flattened_cells = mesh.geometry.dofmap.array
    vert_per_elem = 4 if cell_geom == 'tet' else 3
    cells = flattened_cells.reshape(-1, vert_per_elem)
    return cells

def _ensure_cell_idxs_are_ints(cells, epsilon=1e-6):
    if np.any(np.mod(cells, 1) > epsilon):
        warnings.warn('cells contains floats; these values will be cast to integers.')
    return np.array(cells, dtype=int)

def _check_points_and_cells(coords, cells, cell_geom):
    if coords.ndim != 2:
        raise ValueError(f'Expected points to be a 2d array; instead, it was a {coords.ndim}d array.')
    if (cell_geom == 'tet') & (coords.shape[-1] != 3):
        raise ValueError('Expected points to be an N×3 array, since tetrahedral meshes exist in 3d space;' 
                         f'instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a tetrahedral mesh?')
    if (cell_geom == 'tri') & (cells.shape[-1] != 3):
        raise ValueError('Expected cells to be an N×3 array, since triangular elements consist of three vertices;' 
                         f'instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a triangular mesh?')
    elif (cell_geom == 'tet') & (cells.shape[-1] != 4):
        raise ValueError('Expected cells to be an N×4 array, since tetrahedral elements consist of four vertices;' 
                         f'instead, it was an N×{coords.shape[-1]} array. Are you sure you provided a tetrahedral mesh?')
    if np.min(cells) < 0:
        raise ValueError(f"Expected smallest index in cells to be greater than or equal to 0; instead, the smallest value in cells was {np.min(cells)}.")
    elif np.max(cells) > coords.shape[0]-1:
        raise ValueError(f"Expected largest index in cells to be an int less than or equal to (N-1), where N"
                         f'is the number of vertices listed in points; instead, the largest value in cells was {np.max(cells)},'
                         f'but only N = {coords.shape[0]} vertices were listed in points.')
        
#
#   Subdivision Methods
#

def _subdivide_mesh(points, cells, cell_geom):

    # points.shape = (num_vert, num_spatial_dim)
    # cells.shape = 
    # cell_geom = 'tet' or 'tri'
    
    # Get all possible combinations of local vertices which can be used to construct features (i.e. edges, faces, and olumes)
    local_vert_combos = _create_vert_combos(cell_geom)
    
    # Ensure tets correctly orientated so that local vertex indices are correct (i.e. 'highest' point labelled as vertex 3):
    if cell_geom == 'tet':
        cells = _order_tet_local_verts_by_height(points, cells, local_vert_combos)
    
    # Get global vertex indices of features in each element:
    feature_verts = {feature_name: cells[:,vert_combo] for feature_name, vert_combo in local_vert_combos.items()}
    
    # Iterate over features to create new vertices at centre of features (i.e. mid-edge, mid-face, or mid-vol):
    num_verts = points.shape[0]
    num_new_verts = []
    new_verts = {}
    new_verts_coords = []
    for feature_name, verts in feature_verts.items():
        # feature_name = 'edge', 'face', or 'vol'
        # verts.shape = (num_elems, num_features_per_elem, num_vert_per_feature)
        
        # Step 1: get vertex list of unique features (i.e. vertices/faces/edges) in mesh 
        # List all features in mesh, including repeats:
        reshaped_verts = verts.reshape(-1, verts.shape[-1]) # shape = (num_elems*num_features_per_elem, num_vert_per_feature)
        # Sort vertex numbers within each element:
        sorted_verts = np.sort(reshaped_verts, axis=-1)
        # Remove repeated features (e.g. repeats of edge shared by two neighbouring elements will be removed):
        unique_verts, unique_inv = np.unique(sorted_verts, axis=0, return_inverse=True)
        # unique_verts.shape = (num_unique_features, num_vert_per_feature)
        
        # Step 2: compute coordinates of new vertex points, located at centre of each unique feature (i.e. mid-edge, mid-face, or mid-vol):
        unique_verts_coords = points[unique_verts,:] # shape = (num_unique_features, num_vert_per_feature, num_spatial_dim)
        # Center of feature = average position of vertices in feature
        new_verts_coords.append(np.mean(unique_verts_coords, axis=1)) # shape = (num_unique_features, num_spatial_dim)

        # Step 3: assign number/index to each new vertex.
        # Assign values starting from 0:
        new_vert_idx = np.arange(0, unique_verts.shape[0]) # shape = (num_vert_per_feature,)
        # Shift values up so that they don't overlap with pre-existing vertex numbers:
        new_vert_idx = new_vert_idx + num_verts + sum(num_new_verts) # shape = (num_vert_per_feature,)
        num_new_verts.append(new_vert_idx.size)

        # Step 4: associate each new vertex with the feature and element it was computed from (e.g. all of the new
        # vertices computed from the centre position element 28's faces are all 'grouped' together in an array).
        # Note that this indexing 'brings back' repeated vertices - see 'return_inverse' arg in np.unique documentation:
        grouped_new_verts = new_vert_idx[unique_inv]  # shape = (num_elems*num_features_per_elem,)
        new_verts[feature_name] = grouped_new_verts.reshape(*verts.shape[:-1]) # shape = (num_elems, num_features_per_elem)
    
    # New mesh coords = old mesh coords + coords of new vertices:
    points = np.concatenate([points, *new_verts_coords], axis=0) # shape = (num_verts+sum(num_new_vert), ndim)
    
    # Create cells of new mesh:
    if cell_geom == 'tet':
        cells = _assemble_hexahedrons(feature_verts, new_verts, local_vert_combos) # shape = (4*num_cells, num_vert_per_cell)
    else:
        cells = _assemble_quadrilaterals(feature_verts, new_verts, local_vert_combos) # shape = (2*num_cells, num_vert_per_cell)
    
    return points, cells

def _create_vert_combos(cell_geom):
    vert_per_cell = 4 if cell_geom == 'tet' else 3
    # Edge defined by 2 vertices, face defined by 3 vertices:
    # e.g. if vert_per_cell = 3, then vert_combos['edge'] = [(0,1), (0,2), (1,2)]
    vert_combos = {key: list(combinations(range(vert_per_cell), num_vert)) for key, num_vert in {'edge': 2, 'face': 3}.items()}
    if cell_geom == 'tet':
        # Tetrahedral volume defined by 4 vertices:
        vert_combos['vol'] = list(combinations(range(vert_per_cell), 4))
    return vert_combos

def _order_tet_local_verts_by_height(points, cells, local_vert_combos):
    # Get global indices of vertices which comprise each tet:
    vert_idx = cells[:,local_vert_combos['vol']] # shape = (num_tet, num_vol_per_tet=1, num_vert_per_vol=4)
    # Get coordinates of these vertices:
    verts_coords = points[vert_idx,:] # shape = (num_tet, num_vol_per_tet=1, num_vert_per_vol=4, num_spatial_dim=3)
    # Sort vertices in each element by their z height:
    vert_z_pos = verts_coords[:,0,:,-1]
    z_sort = np.argsort(vert_z_pos, axis=-1)
    cells = np.take_along_axis(cells, z_sort, axis=-1)
    return cells

def _assemble_hexahedrons(feature_verts, new_verts, local_vert_combos):
    
    # Step 1: Create arrays of new vertex idx associated each feature of each element 
    # verts = {0: array with idx of vert 0 in each cell of mesh, ...}
    num_vert_in_vol = 4
    verts = {f'{i}': feature_verts['vol'][:,0,i] for i in range(num_vert_in_vol)}
    feature_idx = []
    for feature_name in ('edge', 'face', 'vol'):
        local_verts = local_vert_combos[feature_name]
        idx_dict = {_create_local_idx_str(vert_idx): new_verts[feature_name][:,i] for i, vert_idx in enumerate(local_verts)}
        feature_idx.append(idx_dict)
    # edges = {'01': array with idx of new verts in middle of edge 01 in each cell of mesh, ...}
    # faces = {'012': array with idx of new verts in middle of face 012 in each cell of mesh, ...}
    # vols = {'0123': array with idx of new verts in middle of vol 0123 in each cell of mesh, ...}
    edges, faces, vols = feature_idx
    
    # Step 2: initialise array to store new cells
    original_num_cell = feature_verts['vol'].shape[0]
    num_vert_per_hex = 8
    num_hex_per_cell = hpc = 4
    cells = np.zeros((num_hex_per_cell*original_num_cell, num_vert_per_hex))
    
    # Step 3: assemble each hexahedron with new + old vertices:  
    # Hexahedron 0 = Vert_0, Edge_01, Face_012, Edge_02, Edge_03, Face_013, Vol_0123, Face_023
    cells[0::hpc,:] = np.stack([verts['0'], edges['01'], faces['012'], edges['02'],
                                edges['03'], faces['013'], vols['0123'], faces['023']], axis=-1)
    # Hexahedron 1 = Vert_1, Edge_12, Face_012, Edge_01, Edge_13, Face_123, Vol_0123, Face_013
    cells[1::hpc,:] = np.stack([verts['1'], edges['12'], faces['012'], edges['01'],
                                edges['13'], faces['123'], vols['0123'], faces['013']], axis=-1)
    # Hexahedron 2 = Vert_2, Edge_02, Face_012, Edge_12, Edge_23, Face_023, Vol_0123, Face_123
    cells[2::hpc,:] = np.stack([verts['2'], edges['02'], faces['012'], edges['12'],
                                edges['23'], faces['023'], vols['0123'], faces['123']], axis=-1)
    # Hexahedron 3 = Edge_03, Face_013, Vol_0123, Face_023, Vert_3, Edge_13, Face_123, Edge_23
    cells[3::hpc,:] = np.stack([edges['03'], faces['013'], vols['0123'], faces['023'],
                                verts['3'], edges['13'], faces['123'], edges['23']], axis=-1)
    
    return cells
    
def _assemble_quadrilaterals(feature_verts, new_verts, local_vert_combos):
    # Step 1: Create arrays of new vertex idx associated each feature of each element 
    # verts = {0: array with idx of vert 0 in each cell of mesh, ...}
    num_vert_in_face = 3 
    verts = {f'{i}': feature_verts['face'][:,0,i] for i in range(num_vert_in_face)}
    feature_idx = []
    for feature_name in ('edge', 'face'):
        local_verts = local_vert_combos[feature_name]
        idx_dict = {_create_local_idx_str(vert_idx): new_verts[feature_name][:,i] for i, vert_idx in enumerate(local_verts)}
        feature_idx.append(idx_dict)
    # edges = {'01': array with idx of new verts in middle of edge 01 in each cell of mesh, ...}
    # faces = {'012': array with idx of new verts in middle of face 012 in each cell of mesh, ...}
    edges, faces = feature_idx
    
    # Step 2: initialise array to store new cells
    original_num_cell = feature_verts['face'].shape[0]
    num_vert_per_quad = 4
    num_quad_per_cell = qpc = 3
    cells = np.zeros((num_quad_per_cell*original_num_cell, num_vert_per_quad))

    # Step 3: assemble each quadrilateral with new + old vertices:  
    # Quadrilateral 0 = Vert_0, Edge_01, Face_012, Edge_02
    cells[0::qpc,:] = np.stack([verts['0'], edges['01'], faces['012'], edges['02']], axis=-1)
    # Quadrilateral 1 = Vert_1, Edge_12, Face_012, Edge_01
    cells[1::qpc,:] = np.stack([verts['1'], edges['12'], faces['012'], edges['01']], axis=-1)
    # Quadrilateral 2 = Face_012, Edge_12, Vert_2, Edge_02
    cells[2::qpc,:] = np.stack([faces['012'], edges['12'], verts['2'], edges['02']], axis=-1)
    
    return cells
    
def _create_local_idx_str(verts):
    return ''.join(str(val) for val in verts)

#
#   Post-Processing Methods
#

def _create_mesh(points, cells, mesh, output_type, cell_geom):
    
    # Some mesh formats throw errors if cells doesn't consist of ints:
    cells = np.array(cells, dtype=int)
    
    if output_type == 'dict':
        mesh = {'points': points, 'cells': cells}
    
    elif output_type == 'meshio':
        try:
            import meshio
        except ImportError:
            raise ValueError('meshio must be installed to output a meshio mesh.')
        cell_key = 'hexahedron' if cell_geom=='tet' else 'quad'
        mesh = meshio.Mesh(points=points, cells={cell_key: cells})
    
    elif output_type == 'pyvista':
        try:
            import pyvista, vtk
        except ImportError:
            raise ValueError('pyvista and vtk must be installed to output a pyvista mesh.')
        # Pyvista throws error if mesh is 2d:
        if points.shape[-1] < 3:
            points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)
        cell_key = vtk.VTK_HEXAHEDRON if cell_geom=='tet' else vtk.VTK_QUAD
        mesh = pyvista.UnstructuredGrid({cell_key: cells}, points)
    
    elif output_type == 'dolfinx':
        # Hexahedra and quadrilaterals are assigned unique identifying int - See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
        try:
            import dolfinx, mpi4py
        except ImportError:
            raise ValueError('dolfinx and mpi4py must be installed to output a dolfinx mesh.')
        gmsh_mesh_num = 5 if cell_geom=='tet' else 3
        num_spatial_dim = points.shape[-1]
        domain = dolfinx.io.ufl_mesh_from_gmsh(gmsh_mesh_num, num_spatial_dim)
        ufl_cell_geom = dolfinx.cpp.mesh.to_type(str(domain.ufl_cell()))
        cells = cells[:, dolfinx.cpp.io.perm_gmsh(ufl_cell_geom, cells.shape[1])]
        mesh = dolfinx.mesh.create_mesh(mpi4py.MPI.COMM_WORLD, cells, points, domain)
        
    return mesh   