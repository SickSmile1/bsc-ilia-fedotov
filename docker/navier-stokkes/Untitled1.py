#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gmsh

gmsh.initialize()
gmsh.model.add("2D_Canal_with_Obstacle")

# Create 2D geometry (example dimensions)
canal_length, canal_height = 10, 2
obstacle_size = 0.5

# Create canal rectangle
canal = gmsh.model.occ.addRectangle(0, 0, 0, canal_length, canal_height)

# Create obstacle (e.g., a circle)
obstacle_x, obstacle_y = 5, 1  # position of obstacle center
obstacle = gmsh.model.occ.addDisk(obstacle_x, obstacle_y, 0, obstacle_size, obstacle_size)

# Cut obstacle from canal
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
gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

# Generate 2D mesh
gmsh.model.mesh.generate(2)

gmsh.write("canal_with_obstacle_2d.msh")
gmsh.finalize()


# In[3]:


get_ipython().run_line_magic('pinfo', 'gmsh.model.getEntities')


# In[ ]:




