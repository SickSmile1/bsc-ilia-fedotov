import gmsh
import pyvista
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from pathlib import Path
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological,
                        locate_dofs_geometrical, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import VTXWriter, distribute_entity_data, gmshio
from dolfinx.mesh import create_mesh, meshtags_from_entities
from basix.ufl import element as basix_element
from dolfinx.plot import vtk_mesh
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, conditional,le,ge,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate)


def mpi_example(comm):
    return f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}"


def init_mesh(mesh_file,canal_length=5.,canal_height=2.,obstacle_center_x=2.,obstacle_center_y=2.,obstacle_radius=2.5,
              d_in=.05,d_out=.25,b_in=.05,b_out=.25):
    """
    Setup the canal with a rectangular domain and an obstacle, and compute the mass flow rate
    at the beginning, middle, and end of the canal.

    Parameters:
    ----------
    canal_length : float
        Length of the canal (horizontal dimension) in meters. Default is 5.0.
    canal_height : float
        Height of the canal (vertical dimension) in meters. Default is 2.0.
    obstacle_center_x : float
        The x-coordinate position of the obstacle's center in the canal. Default is 4.0.
    obstacle_center_y : float
        The y-coordinate position of the obstacle's center.
        For a wall-attached obstacle, this would be `H`. Default is 5.0.
    obstacle_radius : float
        Radius of the circular obstacle inside the canal. Default is 2.5.

    Returns:
    --------
        None
        Creates: .msh file
    --------
    """

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("Rectangle_with_Obstacle_OCC")
    # Create the rectangle (using gmsh.model.occ)
    canal = gmsh.model.occ.addRectangle(0, 0, 0, canal_length, canal_height)
    
    # Create the circular obstacle (using gmsh.model.occ)
    obstacle = gmsh.model.occ.addDisk(obstacle_center_x, obstacle_center_y, 0, obstacle_radius, obstacle_radius)
    
    # Cut the circular obstacle out of the rectangular surface
    fluid_domain, _ = gmsh.model.occ.cut([(2, canal)], [(2, obstacle)])

    gmsh.model.occ.synchronize()
    
    # Define physical groups
    fluid_marker = 1
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    
    # Fluid domain (2D)
    gmsh.model.addPhysicalGroup(2, [fluid_domain[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(2, fluid_marker, "Fluid")
    
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []
    # Define a tolerance for comparing coordinates
    tol = 1e-6
    '''
    points = gmsh.model.getEntities(0)  # Get all points
    curves = gmsh.model.getEntities(1)  # Get all curves
    surfaces = gmsh.model.getEntities(2)  # Get all surfaces
    volumes = gmsh.model.getEntities(3)  # Get all volumes
    '''
    volumes = gmsh.model.getEntities(3)
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    # Iterate over each boundary and check its associated nodes' positions
    for boundary in boundaries:
        # Get the node tags for this boundary
        node_tags = gmsh.model.mesh.getNodes(boundary[0], boundary[1])[0]
        node_coords = gmsh.model.mesh.getNode(node_tags[0])[0]  # Get the first node's coordinates
    
        # Check if the boundary is the inflow (x = 0)
        if np.isclose(node_coords[0], 0.0, atol=tol):
            inflow.append(boundary[1])
        # Check if the boundary is the outflow (x = L)
        elif np.isclose(node_coords[0], canal_length, atol=tol):
            outflow.append(boundary[1])
        # Check if the boundary is a wall (y = 0 or y = H)
        elif np.isclose(node_coords[1], 0.0, atol=tol) or np.isclose(node_coords[1], canal_height, atol=tol):
            walls.append(boundary[1])
        # If none of the above, consider it part of the obstacle
        else:
            obstacle.append(boundary[1])
    # Assign physical groups to the identified boundaries
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")  

    # --------------------------------------------------------------
    # Define mesh fields for a finer mesh around the obstacle
    # --------------------------------------------------------------
    
    # 1. Refine the horizontal part of the rectangle at the obstacle x-coordinate
    gmsh.model.mesh.field.add("Ball", 1)
    gmsh.model.mesh.field.setNumber(1, "Radius", obstacle_radius*1.3)  # Define a circular region of radius 2 units
    gmsh.model.mesh.field.setNumber(1, "XCenter", obstacle_center_x)  # X coordinate of the obstacle
    gmsh.model.mesh.field.setNumber(1, "YCenter", obstacle_center_y)  # Y coordinate (top wall)
    gmsh.model.mesh.field.setNumber(1, "VIn", d_in)   # Minimum element size inside the circular region
    gmsh.model.mesh.field.setNumber(1, "VOut", d_out)   # Maximum element size outside the circular region
    
    # 2. Refine the vertical part of the rectangle where the obstacle is located
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "VIn", b_in)  # Smaller mesh size inside the box
    gmsh.model.mesh.field.setNumber(2, "VOut", b_out)  # Coarser mesh outside the box
    gmsh.model.mesh.field.setNumber(2, "XMin", obstacle_center_x - obstacle_radius)  # Start of the vertical region
    gmsh.model.mesh.field.setNumber(2, "XMax", obstacle_center_x + obstacle_radius)  # End of the vertical region
    gmsh.model.mesh.field.setNumber(2, "YMin", 0)    # Bottom of the rectangle
    gmsh.model.mesh.field.setNumber(2, "YMax", canal_height)  # Full height of the rectangle
    # --------------------------------------------------------------
    # Set the background field to use the obstacle refinement
    # --------------------------------------------------------------
    gmsh.model.mesh.field.add("Min", 3)  # Combine fields to ensure the smallest element size is used
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])
    
    # Set the combined field as the background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    # --------------------------------------------------------------
    # Generate the mesh
    # --------------------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write(mesh_file)

def print_msh(file, name):
    gmsh.open(file)
    gmsh.model.occ.synchronize()
    node_tags, nodes, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(nodes[1]).reshape((-1, 3))
    _, _, elements = gmsh.model.mesh.getElements()
    element_nodes = elements.reshape(-1, 3)
    # Plot the mesh
    plt.figure(figsize=(8, 6))
    for element in element_nodes:
        triangle = node_coords[element, :2]  # Extract x, y coordinates of triangle's nodes
        polygon = plt.Polygon(triangle, edgecolor='black', fill=None)  # Create a polygon
        plt.gca().add_patch(polygon)

    plt.xlim(0, canal_length)
    plt.ylim(0, canal_height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Mesh Visualization of the Rectangle with Obstacleread_gmsh")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(name, format='pdf', bbox_inches="tight")



def calc_with_dolfin(filename,comm):
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename,comm, 0,gdim=2)
    # Get the function space (for velocity in this case, assuming it's a flow problem)
    t = 0
    v_cg2 = basix_element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    s_cg1 = basix_element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    
    # define test_functions for FEM
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

 
    # Define strain-rate tensor
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p, mu):
        return 2 * mu * epsilon(u) - p * Identity(len(u))

    # functions to set dirichlet boundary conditions and inflow p
    def walls(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], canal_height))

    def inflow(x):
        return np.isclose(x[0], 0)

    def outflow(x):
        return np.isclose(x[0], canal_length)

    def u_exact(x):
        L = canal_length  # Length of the canal
        H = canal_height   # Height of the canal
        R = obstacle_radius # Radius of the semi-circular obstacle
        x0 = obstacle_center_x  # x-coordinate of the obstacle center
        y0 = obstacle_center_y  # y-coordinate of the obstacle center (top wall)
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        
        # Calculate distance from obstacle center
        r = np.sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        
        # x-component of velocity
        values[0] = 4 * x[1] * (H - x[1]) / H**2 * (1 - np.exp(-x[0]/L)) * (1 - R**2 / (r**2 + 1e-10))
        
        # y-component of velocity (perturbation due to obstacle)
        values[1] = -0.1 * R**2 * (x[1]-y0) / (r**2 + 1e-10) * (x[0] - x0) / (np.abs(x[0] - x0) + 1e-10)
        
        # Set velocity to zero inside the obstacle
        inside_obstacle = (r < R) & (x[1] > y0 - R)
        values[0][inside_obstacle] = 0
        values[1][inside_obstacle] = 0
        
        return values

    wall_dofs = locate_dofs_geometrical(V, walls)
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip = dirichletbc(u_noslip, wall_dofs, V)

    inflow_dofs = locate_dofs_geometrical(Q, inflow)
    bc_inflow = dirichletbc(PETSc.ScalarType(pressure), inflow_dofs, Q)

    outflow_dofs = locate_dofs_geometrical(Q, outflow)
    bc_outflow = dirichletbc(PETSc.ScalarType(pressure-.5), outflow_dofs, Q)
    bcu = [bc_noslip]
    bcp = [bc_inflow, bc_outflow]

    u_n = Function(V)
    u_n.name = "u_n"
    U = 0.5 * (u_n + u)
    n = FacetNormal(mesh)
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(.001))
    rho = Constant(mesh, PETSc.ScalarType(1))

    # Define the variational problem for the first step
    p_n = Function(Q)
    p_n.name = "p_n"
    F1 = rho * dot((u - u_n) / k, v) * dx
    F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    F1 += inner(sigma(U, p_n, mu), epsilon(v)) * dx
    F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
    F1 -= dot(f, v) * dx

    a1 = form(lhs(F1))
    L1 = form(rhs(F1))

    A1 = assemble_matrix(a1, bcs=bcu) # type: ignore
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
    # play with accuracy of solvers, these are the defail values
    # solver1.setTolerances(rtol=1e-5, atol=1e-50, divtol=1e5, max_it=1000)
    folder = Path("results")
    folder.mkdir(exist_ok=True, parents=True)
    vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
    vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
    vtx_u.write(t)
    vtx_p.write(t)

    u_ex = Function(V)
    u_ex.interpolate(u_exact)

    L2_error = form(dot(u_ - u_ex, u_ - u_ex) * dx)


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
    return [u_n, p_n,V,mesh]

def calculate_flow_and_pressure(u, p, V, mesh):
    # Get the minimum and maximum x-coordinates of the mesh
    x_min = .15
    x_max = mesh.geometry.x[:, 0].max()
    
    # Define the interval for calculations
    interval = 0.1
    
    # Create lists to store the results
    flow_results = []
    pressure_results = []
    
    # Extract the normal component of velocity (u_x in 2D)
    u_sub = u.sub(0)
    
    # Define measures and spatial coordinates
    dx = Measure("dx", domain=mesh)
    x = SpatialCoordinate(mesh)
    
    # Loop through the x-coordinates at regular intervals
    for x_pos in np.arange(x_min, x_max + interval, interval):
        # Define a slice at the current x-coordinate
        slice_condition = conditional(ge(x[0], x_pos - 3e-2), 1.0, 0.0) * conditional(le(x[0], x_pos + 3e-2), 1.0, 0.0)
        
        # Calculate mass flow rate at the current slice
        mass_flow = assemble_scalar(form(u_sub * slice_condition * dx))
        
        # Calculate average pressure at the current slice
        pressure_avg = assemble_scalar(form(p * slice_condition * dx)) / assemble_scalar(form(slice_condition * dx))
        
        # Append the results to the lists
        flow_results.append((x_pos, mass_flow))
        pressure_results.append((x_pos, pressure_avg))
    
    # Communicate results across MPI processes
    if MPI.COMM_WORLD.size > 1:
        flow_results = MPI.COMM_WORLD.allgather(flow_results)
        flow_results = [item for sublist in flow_results for item in sublist]
        pressure_results = MPI.COMM_WORLD.allgather(pressure_results)
        pressure_results = [item for sublist in pressure_results for item in sublist]
    
    # Sort results by x-coordinate
    flow_results.sort(key=lambda x: x[0])
    pressure_results.sort(key=lambda x: x[0])
    
    return flow_results, pressure_results, 

canal_length=2.2
canal_height=.41
obstacle_center_x=1.1
obstacle_center_y=.41
obstacle_radius=.05

if __name__ == "__main__":
    model_rank = 0
    # meshing rate values for disc d and box b 
    d_in, d_out, b_in, b_out = 0.05, 0.25, 0.1, 0.5

    pressure = 8
    t = 0
    T = 6
    num_steps = 1500
    dt = T / num_steps
    comm = MPI.COMM_WORLD
    mpi_example(comm)
    # if comm.rank == model_rank:
    #     # create mesh files
    #     init_mesh("r05.msh",canal_length=2.2,canal_height=.41,obstacle_center_x=1.1,
    #               obstacle_center_y=.41,obstacle_radius=.05,
    #               d_in=.05,d_out=.25,b_in=.05,b_out=.25)
    #     init_mesh("r1.msh",canal_length=2.2,canal_height=.41,obstacle_center_x=1.1,
    #               obstacle_center_y=.41,obstacle_radius=.1,
    #               d_in=.05,d_out=.25,b_in=.05,b_out=.25)
    #     init_mesh("r015.msh",canal_length=2.2,canal_height=.41,obstacle_center_x=1.1,
    #               obstacle_center_y=.41,obstacle_radius=.15,
    #               d_in=.05,d_out=.25,b_in=.05,b_out=.25)
    #     init_mesh("r2.msh",canal_length=2.2,canal_height=.41,obstacle_center_x=1.1,
    #               obstacle_center_y=.41,obstacle_radius=.2,
    #               d_in=.05,d_out=.25,b_in=.05,b_out=.25)
    #     gmsh.finalize()

        # run sims
    u,p,V,mesh = calc_with_dolfin('r05.msh',comm)
    print(calculate_flow_and_pressure(u,p,V,mesh) )

