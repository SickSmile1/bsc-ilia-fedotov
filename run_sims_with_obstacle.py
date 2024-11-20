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
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system,SpatialCoordinate)

if __name__ == "__main__":
    model_rank = 0
    mesh_file = "mesh_with_obstacle_05_25.msh"
    # meshing rate values for disc d and box b 
    d_in, d_out, b_in, b_out = 0.05, 0.25, 0.1, 0.5
    filename = "mfr_l2"

    pressure = 8
    t = 0
    T = 10
    num_steps = 500
    dt = T / num_steps
    comm = MPI.COMM_WORLD
    mpi_example(comm)
        
    if mesh_comm.rank == model_rank:
        init_mesh("r.5.msh",obstacle_radius=.5)
        init_mesh("r1.msh",obstacle_radius=1.)
        init_mesh("r1.5.msh",obstacle_radius=1.5)
        init_mesh("r1.9.msh",obstacle_radius=1.9)

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
        elif np.isclose(node_coords[0], L, atol=tol):
            outflow.append(boundary[1])
        # Check if the boundary is a wall (y = 0 or y = H)
        elif np.isclose(node_coords[1], 0.0, atol=tol) or np.isclose(node_coords[1], H, atol=tol):
            walls.append(boundary[1])
        # If none of the above, consider it part of the obstacle
        else:
            obstacle.append(boundary[1])

    # Assign physical groups to the identified boundaries
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inx = ufl.SpatialCoordinate(V.mesh))

    # --------------------------------------------------------------
    # Define mesh fields for a finer mesh around the obstacle
    # --------------------------------------------------------------
    
    # 1. Refine the horizontal part of the rectangle at the obstacle x-coordinate
    gmsh.model.mesh.field.add("Ball", 1)
    gmsh.model.mesh.field.setNumber(1, "Radius", 3.0)  # Define a circular region of radius 2 units
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

