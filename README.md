# tetridiv

## Overview

`tetridiv` subdivides tetrahedral meshes into hexahedral meshes and triangular meshes into quadrilateral ones. The motivating use-case for `tetridiv` was so that [`dolfinx`](https://github.com/FEniCS/dolfinx) meshes could be easily subdivided, although `tetridiv` is also capable of working with [`meshio`](https://github.com/nschloe/meshio), [`pyvista`](https://github.com/pyvista/pyvista), and [`tetgen`](https://github.com/pyvista/tetgen) meshes. Since all computationally-intensive subdivision operations are vectorised, `tetridiv` should be relatively quick, even for large meshes, despite being written purely in Python.

## Installation

`tetridiv` can be installed using `pip` like so:
```
pip install git+https://github.com/MABilton/tetridiv
```

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
The returned subdivided meshes should be the same type as the input mesh.

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

Please refer to the Jupyter notebooks in the `examples` folder for more specific examples on how to use `tetridiv`.

## Why Subdivide Meshes?

Tetrahedral meshes are convenient since they are easily generated from surface data, even for complex surface geometries; the [`tetgen`](https://github.com/pyvista/tetgen) package provides one such implementation. However, tetrahedral meshes exhibit undesirable behaviours in specific applications. In Finite Element Method simulations of non-linear mechanics, for instance, tetrahedral meshes tend to exhibit 'volumetric locking', which causes them to behave artificially stiffly. Conversely, hexahedral meshes tend not to exhibit this locking phenomenon. For a more thorough description of the volumetric locking phenomenon, please refer to [[1]](#1).

In an ideal world, one can generate a hexahedral mesh from any arbitrarily complicated surface directly; unfortunately, generating a hexahedral mesh from surface data is a highly non-trivial problem with no standard 'silver bullet' solution. One particularly naive solution to this problem is to first create a tetrahedral mesh and then subdivide this tetrahedral mesh into a hexahedral one. Please refer to [[2]](#2) for more details on this.

Although this 'tetrahedralise-then-subdivide' algorithm is simple, I've yet to come across any easy-to-use, Pythonic implementations which can easily interface with other Python packages. `tetridiv` is my attempt to provide a simple, 'no-nonsense' implementation of this algorithm.

## Key Limitations

`tetridiv` is designed only to subdivide tetrahedral and triangular meshes - it's not designed to subdivide any other kinds of mesh.

Additionally, `tetridiv` is currently unable to work with mixed meshes (e.g. meshes composed of tetrahedral and hexahedral elements), although this functionality may be implemented later.

## Docker Container with `dolfinx` and `tetridiv`

The easiest way to run `dolfinx` is by using the [`dolfinx/lab` Docker container](https://hub.docker.com/r/dolfinx/lab), which automatically starts up a Jupyter Lab instance. For the sake of convenience, I created the [`mabilton/tetrigen` Docker image](https://hub.docker.com/repository/docker/mabilton/tetridiv) based on `dolfinx/lab:v0.4.1`; this image contains all of the dependencies required to run the Jupyter notebooks in the `examples` folder. See the `Dockerfile` in this repository for further details.

To start up a container from the `mabilton/tetrigen` image, first [install Docker](https://docs.docker.com/engine/install/), make sure it's running, and then execute:
```
docker run -v "$(pwd)":/root/ -p 8888:8888 mabilton/tetridiv
```
inside of a directory that **contains the Jupyter notebooks** you want to run. Upon executing this command, a Jupyter Lab instance should be accessible at `localhost:8888` in your browser. Using this container, all of the Jupyter notebooks in the `examples` folder should work without further installations.

We'll note here that this Docker image is **not required** to run `tetrdiv` - it should work perfectly fine even when it's not being run inside of a container. Instead, the only real purpose of this image is to provide an environment that has both `dolfinx` and `tetridiv` (along with a few other useful packages) installed.

## Tests

To run the *non-`dolfinx`* related tests, execute:
```
pytest .
```
**inside the `tetrigen` folder**. To run all tests, including those on `dolfinx` meshes, the `mabilton/tetrigen` Docker container must be used; this is most easily done by running the `docker-run-tests.sh` shell script from **inside the `tetrigen` repository**:
```
./docker-run-tests.sh
```
Calling this script will launch a `mabilton/tetrigen` container, perform the tests, and stop and delete the container.

## References

<a id="1">[1]</a>
Tadepalli SC, Erdemir A, Cavanagh PR. *Comparison of hexahedral and tetrahedral elements in finite element analysis of the foot and footwear*. J Biomech. 2011; 44(12):2337-2343. doi:10.1016/j.jbiomech.2011.05.006. [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/21742332/).

<a id="2">[2]</a>
Pietroni N, Campen M, Sheffer A, Cherchi G, Bommes D, Gao X, Scateni R, Ledoux F, Remacle JF, Livesu M. *Hex-Mesh Generation and Processing: a Survey*. arXiv preprint. 2022; arXiv:2202.12670. doi:10.48550/arXiv.2202.12670. [arXiv Link](https://arxiv.org/abs/2202.12670).
