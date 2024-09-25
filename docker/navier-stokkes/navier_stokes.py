# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter, gmshio
from dolfinx.mesh import create_unit_square,create_rectangle, create_mesh, CellType
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym, grad, inner)

# %%
#comm = MPI.COMM_WORLD
#mesh, a1, a2 = gmshio.read_from_msh("mesh/bezier2.msh", comm)
#x1 = mesh.geometry.x
#a1

# %%
comm = MPI.COMM_WORLD
height, length = .55, 5
discrete_x, discrete_y = 50,10
mesh = create_rectangle(MPI.COMM_WORLD, [[0.,0.], [length, height]],[discrete_x,discrete_y], CellType.triangle)
#mesh, _, _ = gmshio.read_from_msh("mesh/bezier1.msh", comm)

pressure = 25 # kpa, due to length in micrometer
t = 0
T = 10
num_steps = 500
dt = T / num_steps

# %%
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# %%
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)


# %%
# functions to set dirichlet boundary conditions and inflow p
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], height))

def inflow(x):
    return np.isclose(x[0], 0)

def outflow(x):
    return np.isclose(x[0], length)

def return_all(x):
    return x


# %%
# %matplotlib inline
wall_dofs = locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, wall_dofs, V)

# %%
inflow_dofs = locate_dofs_geometrical(Q, inflow)
bc_inflow = dirichletbc(PETSc.ScalarType(pressure), inflow_dofs, Q)

# %%
outflow_dofs = locate_dofs_geometrical(Q, outflow)
bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]

# %%
# %matplotlib inline
u_n = Function(V)
u_n.name = "u_n"
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant(mesh, PETSc.ScalarType((0, 0)))
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(1))
rho = Constant(mesh, PETSc.ScalarType(1))


# %%
# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

#a1 = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx], [inner(div(u), q) * dx, None]])
#L1 = form([inner(f, v) * dx, inner(Constant(mesh, PETSc.ScalarType(0)), q) * dx])  # type: ignore
# Define the variational problem for the first step
p_n = Function(Q)
p_n.name = "p_n"
F1 = rho * dot((u - u_n) / k, v) * dx
F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
F1 += inner(sigma(U, p_n), epsilon(v)) * dx
F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
F1 -= dot(f, v) * dx

# %%
a1 = form(lhs(F1))
L1 = form(rhs(F1))

# %%
A1 = assemble_matrix(a1, bcs=bcu) # type: ignore
A1.assemble()
b1 = create_vector(L1)

# %%
# Define variational problem for step 2
u_ = Function(V)
a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# Define variational problem for step 3
p_ = Function(Q)
a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# %%
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# %%
from pathlib import Path
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
vtx_u.write(t)
vtx_p.write(t)


# %%
def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values


u_ex = Function(V)
u_ex.interpolate(u_exact)

L2_error = form(dot(u_ - u_ex, u_ - u_ex) * dx)

# %%
for i in range(num_steps):
    # Update current time step
    t += dt

    # Step 1: Tentative veolcity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # Write solutions to file
    vtx_u.write(t)
    vtx_p.write(t)

    # Compute error at current time-step
    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    error_max = mesh.comm.allreduce(np.max(u_.vector.array - u_ex.vector.array), op=MPI.MAX)
    # Print error only every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")
# Close xmdf file
vtx_u.close()
vtx_p.close()
b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

# %%
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(Q)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
plotter = pyvista.Plotter()

x = np.linspace(0, 1, 561)
chart = pyvista.Chart2D()
chart.line(x, p_.x.array[:])
# chart.x_range = [5, 10]  # Focus on the second half of the curve
chart.show()


# %%
# read: https://en.wikipedia.org/wiki/Pressure-correction_method
import matplotlib.pyplot as plt
topology, cell_types, geometry = vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_n)] = u_n.x.array.real.reshape((geometry.shape[0], len(u_n)))
#values[:,1]
plt.plot(np.linspace(0, len(values[:,1]),len(values[:,1])), values[:,1] )

# %%
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_n)] = u_n.x.array.real.reshape((geometry.shape[0], len(u_n)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.7)

# Create a pyvista-grid for the mesh
#mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
#grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
#plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.window_size = [800, 600]
plotter.set_scale(yscale=4)
plotter.view_xy()
plotter.save_graphic("glyphs2.pdf",title='PyVista Export')
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs2.pdf")

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import locate_dofs_geometrical, dirichletbc
from dolfinx.mesh import create_rectangle, CellType

# Create a rectangular mesh
mesh = create_rectangle(MPI.COMM_WORLD, [[0., 0.], [5., .55]], [50, 10], CellType.triangle)

# Functions to define boundaries
def lower_wall(x):
    return np.isclose(x[1], 0)

def upper_wall(x):
    return np.isclose(x[1], .55)

def inflow(x):
    return np.isclose(x[0], 0)

def outflow(x):
    return np.isclose(x[0], 5)

# Locate DOFs for inflow, outflow, lower and upper walls
inflow_dofs = locate_dofs_geometrical(Q, inflow)
outflow_dofs = locate_dofs_geometrical(Q, outflow)
lower_wall_dofs = locate_dofs_geometrical(Q, lower_wall)
upper_wall_dofs = locate_dofs_geometrical(Q, upper_wall)

# Extract points from the mesh
points = mesh.geometry.x

# Plot the mesh and boundary regions using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the full mesh
for cell in mesh.topology.connectivity(2, 0).array.reshape(-1, 3):
    triangle = points[cell]
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.5)

# Plot inflow boundary points
inflow_points = points[inflow_dofs]
ax.plot(inflow_points[:, 0], inflow_points[:, 1], 'bo', label="Inflow")

# Plot outflow boundary points
outflow_points = points[outflow_dofs]
ax.plot(outflow_points[:, 0], outflow_points[:, 1], 'ro', label="Outflow")

# Plot lower wall points
lower_wall_points = points[lower_wall_dofs]
ax.plot(lower_wall_points[:, 0], lower_wall_points[:, 1], 'go', label="Lower Wall")

# Plot upper wall points
upper_wall_points = points[upper_wall_dofs]
ax.plot(upper_wall_points[:, 0], upper_wall_points[:, 1], 'mo', label="Upper Wall")

# Add labels, legend and show the plot
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.tight_layout()
ax.set_title('Mesh with Inflow, Outflow, Lower and Upper Walls')
ax.legend()
#ax.set_aspect('equal')
plt.show()

# %%
