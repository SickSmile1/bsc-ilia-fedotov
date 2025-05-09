# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
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

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction, Measure,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)
import dolfinx

# %%
import gmsh
from dolfinx.io import gmshio

# %%
import matplotlib.pyplot as plt

# %%
import logging

# %%
logging.basicConfig(level=logging.INFO)

# %%
logger = logging.getLogger()


# %%
def create_mesh(comm, H=1, L=3, lc=.07, filename="mesh.msh"):
    """
    Create a 2D mesh of a canal using GMSH.

    This function generates a mesh for a rectangular canal. It uses GMSH for mesh generation and MPI for parallel processing.

    :param comm: MPI communicator
        The MPI communicator object for parallel processing.
    :param H: float, optional
        Height of the canal (default is 1).
    :param L: float, optional
        Length of the canal (default is 3).
    :param lc: float, optional
        Characteristic length for mesh generation (default is 0.07).

    :return: tuple
        A tuple containing:
        - mesh: dolfinx.mesh.Mesh
            The generated mesh.
        - ct: dolfinx.mesh.MeshTags
            Cell tags for the mesh.
        - ft: dolfinx.mesh.MeshTags
            Facet tags for the mesh.
        - infl: int
            Tag for the inflow boundary.
        - outfl: int
            Tag for the outflow boundary.
        - upper: int
            Tag for the upper wall boundary.
        - lower: int
            Tag for the lower wall boundary.

    :raises: None explicitly, but may raise exceptions from GMSH or MPI operations.

    .. note::
        - This function uses GMSH for mesh generation and dolfinx for mesh representation.
        - The mesh is generated only on the process with rank 0 and then broadcast to all processes.
        - The function defines physical groups for different parts of the geometry (inflow, outflow, walls).

    .. warning::
        Ensure that GMSH and dolfinx are properly installed and imported in your environment.

    Example:
        >>> from mpi4py import MPI
        >>> comm = MPI.COMM_WORLD
        >>> mesh, ct, ft, infl, outfl, upper, lower = create_obst(comm)
    """
    model_rank = 0
    infl, outfl, upper, lower = [],[],[],[]
    if comm.rank == model_rank:
        gmsh.initialize()
        gmsh.model.add("canal")
        
        cm = 1 # e-02 # not needed for our sim
        h1 = H * cm
        l1 = L * cm
        Lc1 = lc
        
        # We start by defining some points and some lines. To make the code shorter we
        # can redefine a namespace:
        factory = gmsh.model.geo
        model = gmsh.model
        
        p_lower_left = factory.addPoint(0, 0, 0, Lc1)
        p_lower_right = factory.addPoint(l1, 0, 0, Lc1)
        p_upper_right = factory.addPoint(l1, h1 , 0, Lc1)
        p_upper_left = factory.addPoint(0, h1, 0, Lc1)
        
        l_lower = factory.addLine(p_lower_left, p_lower_right)
        l_right = factory.addLine(p_lower_right, p_upper_right)
        l_upper = factory.addLine(p_upper_right, p_upper_left)
        l_left = factory.addLine(p_upper_left, p_lower_left)
         
        lines = [l_lower,l_right, l_upper, l_left]
        
        # Define the inner curve loop (the circle arc)
        loop = factory.addCurveLoop(lines)
    
        # Create the plane surface with a hole
        surface = factory.addPlaneSurface([loop])
        factory.synchronize()
        
        upper = model.addPhysicalGroup(1, [l_upper], tag=1, name="upper_wall")
        outfl = model.addPhysicalGroup(1, [l_right], tag=2, name="outfl")
        infl = model.addPhysicalGroup(1, [l_left], tag=3, name="inflow")
        lower = model.addPhysicalGroup(1, [l_lower], tag=4, name="lower_wall")

        gmsh.model.addPhysicalGroup(2, [surface], tag=5, name="Domain")
        
        factory.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        
    # infl = comm.bcast(infl, root=0)
    # outfl = comm.bcast(outfl, root=0)
    # upper = comm.bcast(upper, root=0)
    # lower = comm.bcast(lower, root=0)
    # gmsh.model = comm.bcast(gmsh.model, root=0)
    mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
    return mesh, ct, ft, infl, outfl, upper, lower


# %%
mesh, ct, ft, inlet_marker,outlet_marker, upper_wall_marker, lower_wall_marker = create_mesh(MPI.COMM_WORLD, H=1, L=10, lc=0.05)

# %%
upper_wall_marker

# %%
T = 2
# num_steps = 50
dt = 0.02 # T / num_steps
num_steps = int(T/dt)
relative_tolerance = 1.e-5

# %%
final_mass_flow_list = []
pressure_list = []

# %%
for j, pressure in enumerate(np.arange(600, 800, 20)):
    # mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    pressure_list.append(pressure)
    t = 0
    
    logger.info("RUN %d: pressure %d", j, pressure)

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    
    # def walls(x):
    #    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
    
    # wall_dofs = locate_dofs_geometrical(V, walls)
    # u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    # bc_noslip = dirichletbc(u_noslip, wall_dofs, V)
    
    # def inflow(x):
    #    return np.isclose(x[0], 0)
    
    # inflow_dofs = locate_dofs_geometrical(Q, inflow)
    # bc_inflow = dirichletbc(PETSc.ScalarType(pressure), inflow_dofs, Q)
    
    # def outflow(x):
    #    return np.isclose(x[0], 1)
    
    # outflow_dofs = locate_dofs_geometrical(Q, outflow)
    # bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
    # bcu = [bc_noslip]
    # bcp = [bc_inflow, bc_outflow]

    fdim = mesh.topology.dim - 1

    upper_wall_dofs = locate_dofs_topological(V, fdim, ft.find(upper_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip1 = dirichletbc(u_noslip, upper_wall_dofs, V)

    lower_wall_dofs = locate_dofs_topological(V, fdim, ft.find(lower_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip2 = dirichletbc(u_noslip, lower_wall_dofs, V)
    
    inflow_dofs = locate_dofs_topological(Q, fdim, ft.find(inlet_marker))
    bc_inflow = dirichletbc(PETSc.ScalarType(pressure), inflow_dofs, Q)
    
    outflow_dofs = locate_dofs_topological(Q, fdim, ft.find(outlet_marker))
    bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
    bcu = [bc_noslip1, bc_noslip2]
    bcp = [bc_inflow, bc_outflow]
    
    u_n = Function(V)
    u_n.name = "u_n"
    U = 0.5 * (u_n + u)
    n = FacetNormal(mesh)
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(1))
    rho = Constant(mesh, PETSc.ScalarType(1))
    
    # Define strain-rate tensor
    def epsilon(u):
        return sym(nabla_grad(u))
    
    # Define stress tensor
    def sigma(u, p):
        return 2 * mu * epsilon(u) - p * Identity(len(u))
    
    # Define the variational problem for the first step
    p_n = Function(Q)
    p_n.name = "p_n"
    F1 = rho * dot((u - u_n) / k, v) * dx
    F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    F1 += inner(sigma(U, p_n), epsilon(v)) * dx
    F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
    F1 -= dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    
    A1 = assemble_matrix(a1, bcs=bcu)
    A1.assemble()
    b1 = create_vector(L1)
    
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
    
    from pathlib import Path
    folder = Path(f"results/{pressure}")
    folder.mkdir(exist_ok=True, parents=True)
    vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
    vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
    vtx_u.write(t)
    vtx_p.write(t)
    
    def u_exact(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * x[1] * (1.0 - x[1])
        return values
    
    
    u_ex = Function(V)
    u_ex.interpolate(u_exact)
    
    L2_error = form(dot(u_ - u_ex, u_ - u_ex) * dx)
    
    left_boundary = ft.find(inlet_marker)
    right_boundary = ft.find(outlet_marker)
    upper_boundary = ft.find(upper_wall_marker)
    lower_boundary = ft.find(lower_wall_marker)
    
    logger.info("facets with tag 3: %s", left_boundary)
    logger.info("number of facets with tag 3: %s", len(left_boundary))
    
    logger.info("facets with tag 2: %s", right_boundary)
    logger.info("number of facets with tag 2: %s", len(right_boundary))
    
    logger.info("facets with tag 1: %s", upper_boundary)
    logger.info("number of facets with tag 1: %s", len(upper_boundary))
    
    logger.info("facets with tag 4: %s", lower_boundary)
    logger.info("number of facets with tag 4: %s", len(lower_boundary))
    
    # create new MeshTags object to mark rough boundary
    left_boundary_facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, left_boundary, np.full_like(left_boundary, 1, dtype=np.int32))
    right_boundary_facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, right_boundary, np.full_like(right_boundary, 1, dtype=np.int32))
    upper_boundary_facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, upper_boundary, np.full_like(upper_boundary, 1, dtype=np.int32))
    lower_boundary_facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, lower_boundary, np.full_like(lower_boundary, 1, dtype=np.int32))
    
    left_boundary_ds = Measure("ds", domain=mesh, subdomain_data=left_boundary_facet_tags)
    right_boundary_ds = Measure("ds", domain=mesh, subdomain_data=right_boundary_facet_tags)
    upper_boundary_ds = Measure("ds", domain=mesh, subdomain_data=upper_boundary_facet_tags)
    lower_boundary_ds = Measure("ds", domain=mesh, subdomain_data=lower_boundary_facet_tags)
    
    # x = SpatialCoordinate(mesh)
    # n = FacetNormal(mesh)
    mfl = dot(u_n, n)
    # mfl = u_n[0]
    
    mfl_left_expression = form(mfl*left_boundary_ds)
    mfl_right_expression = form(mfl*right_boundary_ds)
    mfl_upper_expression = form(mfl*upper_boundary_ds)
    mfl_lower_expression = form(mfl*lower_boundary_ds)
    
    mass_flow_left_list = []
    mass_flow_right_list = []
    
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    
    N_points = 100
    
    y_grid = np.linspace(0, 1, N_points)
    
    points = np.zeros((3, N_points))
    points[0, :] = 0.5
    points[1, :] = y_grid
    
    # Find cells whose bounding-box collide with the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    
    # numerical integaration along cross-section
    points_on_proc = []
    cells = []
    
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            logger.warning("Point %d: %s not in domain", i, point)
    
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    
    velocity_profiles = []
    
    mass_flow_through_cross_section_list = []

    time_list = []
    
    for i in range(num_steps):
        # Update current time step
        t += dt

        time_list.append(t)
        
        # Step 1: Tentative veolcity step
        with b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_.x.petsc_vec)
        u_.x.scatter_forward()
    
        # Step 2: Pressure corrrection step
        with b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, p_.x.petsc_vec)
        p_.x.scatter_forward()
    
        # Step 3: Velocity correction step
        with b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()
        # Update variable with solution form this time step
        u_n.x.array[:] = u_.x.array[:]
        p_n.x.array[:] = p_.x.array[:]
    
        # def mfl_press_on_boundary(comm,x_max, mesh, mesh_tag, u_n, p):    
        # Define measures and spatial coordinates
        # if i != 0:
        mfl_left_local = assemble_scalar(form(mfl_left_expression))
        mass_flow_left = mesh.comm.allreduce(mfl_left_local, op=MPI.SUM)
    
        mfl_right_local = assemble_scalar(form(mfl_right_expression))
        mass_flow_right = mesh.comm.allreduce(mfl_right_local, op=MPI.SUM)
    
        mfl_upper_local = assemble_scalar(form(mfl_upper_expression))
        mass_flow_upper = mesh.comm.allreduce(mfl_upper_local, op=MPI.SUM)
    
        mfl_lower_local = assemble_scalar(form(mfl_lower_expression))
        mass_flow_lower = mesh.comm.allreduce(mfl_lower_local, op=MPI.SUM)
    
        velocity_profile = u_n.eval(points_on_proc, cells).T
    
        mass_flow_through_cross_section = np.trapz(y=velocity_profile[0], x=y_grid)
        
        velocity_profiles.append(velocity_profile)
        mass_flow_through_cross_section_list.append(mass_flow_through_cross_section)
        
        # print("mass_flow ds: ",mfl_local) # , "pressure: ", pressure_avg)
        mass_flow_left_list.append(mass_flow_left)
        mass_flow_right_list.append(mass_flow_right)

        vtx_u.write(t)
        vtx_p.write(t)
        # Compute error at current time-step
        # error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
        # error_max = mesh.comm.allreduce(np.max(u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
        # Print error only every 20th step and at the last step
        if (i % 20 == 0) or (i == num_steps - 1):
            logger.info(
                f"Time {t:.2f}, Flux left {mass_flow_left:.2e}, Flux right {mass_flow_right:.2e}, Flux upper {mass_flow_upper:.2e}, Flux lower {mass_flow_lower:.2e}, Flux center {mass_flow_through_cross_section:.2e}")        

        if len(mass_flow_through_cross_section_list) > 1:
            mean_of_last_two = np.mean(mass_flow_through_cross_section_list[-2:])
            dm = np.abs(mass_flow_through_cross_section_list[-1] - mass_flow_through_cross_section_list[-2])
            relative_deviation = dm / mean_of_last_two
            logger.debug("Absolute deviation: %g", dm)
            logger.debug("Relative deviation: %g", relative_deviation)
            if relative_deviation < relative_tolerance:
                logger.info("Relative mass flow change %g converged within relatvie tolerance %g", relative_deviation, relative_tolerance)
                break
            
    # Close xmdf file
    vtx_u.close()
    vtx_p.close()
    b1.destroy()
    b2.destroy()
    b3.destroy()
    solver1.destroy()
    solver2.destroy()
    solver3.destroy()

    # np.array(velocity_profile)
    cross_section_data = np.vstack([points[1,:], velocity_profile[0]])
    np.savetxt(folder / "cross_section_velocity_profile.csv", cross_section_data.T, delimiter=",", header="position, velocity")

    mass_flow_evolution = np.vstack([np.array(time_list), np.array(mass_flow_through_cross_section_list)])
    np.savetxt(folder / "mass_flow_evolution.csv", mass_flow_evolution.T, delimiter=",", header="time, flux")

    final_mass_flow_list.append(mass_flow_through_cross_section)

pressure_mass_flow_releation = np.vstack([np.array(pressure_list), np.array(final_mass_flow_list)])
np.savetxt("results/pressure_mass_flow_releation.csv", pressure_mass_flow_releation.T, delimiter=",", header="pressure, flux")

# %%
plt.plot(pressure_mass_flow_releation[0], pressure_mass_flow_releation[1])

# %%
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_n)] = u_n.x.array.real.reshape((geometry.shape[0], len(u_n)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

# Create a pyvista-grid for the mesh
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")

# %%
