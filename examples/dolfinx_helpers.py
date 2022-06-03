import os

import dolfinx
import numpy as np
import ufl
import utils
from mpi4py import MPI
from petsc4py import PETSc


def simulate_neohookean_material(
    mesh,
    C_1,
    density,
    g,
    kappa,
    elem_order,
    rtol,
    atol,
    max_iter,
    num_load_steps,
    y_rot=0,
    x_rot=0,
    **ignored_kwargs,
):
    _clear_fenics_cache()
    # Create function space:
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(mesh, V)
    u = dolfinx.fem.Function(V)
    B = dolfinx.fem.Constant(mesh, [0.0, 0.0, 0.0])
    F = _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, u, B)
    # Define problem:
    solver = _create_nonlinear_solver(F, u, V, bcs, rtol, atol, max_iter)
    f = _create_load_vector(y_rot, x_rot, density, g)
    u = _perform_load_stepping(solver, u, B, f, num_load_steps)
    return u


def _clear_fenics_cache(cache_dir="/root/.cache/fenics"):
    if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
        os.system(f"rm -r {cache_dir}/*")


def _create_clamped_bcs(mesh, V):
    def fixed(x):
        return np.isclose(x[0], 0)

    fixed_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, fixed
    )
    facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, fixed_facets, 1)
    u_bc = dolfinx.fem.Function(V)
    with u_bc.vector.localForm() as loc:
        loc.set(0)
    left_dofs = dolfinx.fem.locate_dofs_topological(
        V, facet_tag.dim, facet_tag.indices[facet_tag.values == 1]
    )
    bcs = [dolfinx.fem.dirichletbc(u_bc, left_dofs)]
    return bcs


def _create_load_vector(y_rot, x_rot, density, g, g_dir=(1, 0, 0)):
    f = density * g * np.array(g_dir)
    f = utils.rotate_vector(f, y_rot, x_rot)
    return f


def _create_nonlinear_solver(F, u, V, bcs, rtol, atol, max_iter):
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = rtol
    solver.atol = atol
    solver.max_it = max_iter
    return solver


def _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, u, B, quad_degree=4):
    v = ufl.TestFunction(V)
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    J = ufl.variable(ufl.det(F))
    Ic = ufl.variable(ufl.tr(C))
    # Nearly-Incompressible Neo-Hookean material;
    # See: https://link.springer.com/article/10.1007/s11071-015-2167-1
    psi = C_1 * (Ic - 3) + kappa * (J - 1) ** 2
    P = ufl.diff(psi, F)
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", metadata=metadata)
    Pi = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx
    return Pi


def _perform_load_stepping(solver, u, B, f, num_load_steps):
    for step_i in range(num_load_steps):
        print(f"Performing load step {step_i+1}/{num_load_steps}...")
        for j, f_j in enumerate(f):
            B.value[j] = ((step_i + 1) / num_load_steps) * f_j
        num_its, converged = solver.solve(u)
        if not converged:
            raise ValueError(f"Load step {step_i+1} failed to converge.")
        u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
    return u
