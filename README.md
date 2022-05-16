# tetridiv

## Overview

`tetridiv` subdivides tetrahedral meshes into hexahedral meshes and triangular meshes into quadrilateral ones. The motivating use-case for `tetridiv` was so that [`dolfinx`](https://github.com/FEniCS/dolfinx) meshes could be easily subdivided, although `tetrigen` is also capable of working with [`meshio`](https://github.com/nschloe/meshio), [`pyvista`](https://github.com/pyvista/pyvista), and [`tetgen`](https://github.com/pyvista/tetgen) meshes. Since all of the computationally-intensive subdivision operations are vectorised, `tetrigen` should be relatively quick, even for large meshes, despite being written purely in Python.

## Example Usage

`tetridiv` provides two functions: `tet2hex`, which subdivides tetrihedral meshes into hexahedral ones, and `tri2quad`, which subdivides triangular meshes into quadrilateral ones:
```python
from tetridiv import tet2hex, tri2quad
```
Once imported, `tet2hex` and `tri2quad` can be used to subdivide meshes of supported formats (i.e. `meshio`, `pyvista`, `tetgen` and `dolfinx`):
```python
hexahedral_mesh = tet2hex(mesh=tetrahedral_mesh)
quadrilateral_mesh = tri2quad(mesh=trangular_mesh)
```
The returned subdivided meshes should be of the same type as the input mesh.

If your mesh is not of a supported type, you can explicitly specify the coordinates and connectivity of your mesh using the `points` and `cells` keyword arguments respectively:
```python
hexahedral_mesh = tet2hex(points=tetrahedral_points, cells=tetrahedral_cells)
quadrilateral_mesh = tri2quad(points=triangular_points, cells=triangular_cells)
```
In this case:
- `tetrahedral_points` is an `(N,3)` array of vertex coordinates, where `N` is the number of vertices in the mesh
- `triangular_points` is an `(N,2)` (if the mesh is 2D) or an  `(N,3)` (if the mesh is 3D) array of vertex coordinates, where `N` is the number of vertices in the mesh
- `tetrahedral_cells` is an `(M,4)` array whose `i'th` row lists the vertex indices which make up the `i'th` tetrahedral in the mesh, where M is the number of tetrahedral elements in the mesh. The order of the vertices should *not* affect the outputted mesh.
- `triangular_cells` is an `(M,3)` array whose `i'th` row lists the vertex indices which make up the `i'th` triangle in the mesh, where M is the number of triangular elements in the mesh. The order of the vertices should *not* affect the outputted mesh.

Please refer to the Jupyter notebooks in the `examples` folder for more specific examples on how to use `tetrigen`.

## Installation

`tetridiv` can be installed through `pip` like so:
```
pip install git+https://github.com/MABilton/tetridiv
```

## Why Subdivide Meshes?

Tetrahedral mesh are convenient since they can easily be generated from surface data, even for complex surface geometries; the [`tetgen`](https://github.com/pyvista/tetgen) package provides one such implementation. However, tetrahedral meshes exhibit undesirable behaviours in certain applications. In Finite Element Method simulations of non-linear mechanics for instance, tetrahedral meshes tend to exhibit 'volumetric locking', which causes them to behave in an artificially stiff manner. Conversely, hexahedral meshes tend not to exhibit this locking phenonemenon. For a more thorough description of the volumteric locking phenomenon, please refer to [[1]](#1).

In an ideal world, one would be; unfortunately, generating a hexahedral mesh from surface data is a highly non-trivial problem with no standard 'silver bullet' solution. One particularly naive solution to this problem is to first create a tetrahedral mesh and then subdivide this tetrahedral mesh into a hexahedral one. Please refer to [[2]](#2) for more details on this.

Although this 'tetrahedralise-then-subdivide' algorithm is simple, I've yet to come across any easy-to-use, Pythonic implementationswhich easily interfaces with other Python packages. `tetrigen` is my attempt to provide a simple, 'no-nonsense' implementation of this algorithm.

## Key Limitations

`tetridiv` is designed only to subdivide tetahedral and triangular meshes - it's not designed to subdivide any other kinds of mesh.

Additionally, `tetridiv` is curently unable to work with mixed meshes (e.g. meshes composed of tetrahedral and hexahedral elements), although this functionality may be implemented at a later date.

## References

<a id="1">[1]</a> 
Tadepalli SC, Erdemir A, Cavanagh PR. *Comparison of hexahedral and tetrahedral elements in finite element analysis of the foot and footwear*. J Biomech. 2011; 44(12):2337-2343. doi:10.1016/j.jbiomech.2011.05.006. [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/21742332/).

<a id="2">[2]</a> 
Pietroni N, Campen M, Sheffer A, Cherchi G, Bommes D, Gao X, Scateni R, Ledoux F, Remacle JF, Livesu M. *Hex-Mesh Generation and Processing: a Survey*. arXiv preprint. 2022; arXiv:2202.12670. doi:10.48550/arXiv.2202.12670. [arXiv Link](https://arxiv.org/abs/2202.12670).
