"""
tetridiv.

A light-weight package that provides:
  1. The `tet2hex` function, which subdivides tetrahedral meshes into hexahedral meshes.
  2. The `tri2quad` function, which subdivides triangular meshes into quadrilateral meshes.

See help(tet2hex) and help(tri2quad) for further details and examples.
"""

from .tetridiv import tet2hex, tri2quad

__all__ = ["tet2hex", "tri2quad"]
__version__ = "0.0.1"
