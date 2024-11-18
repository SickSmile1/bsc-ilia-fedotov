from mpi4py import MPI
from petsc4py import PETSc

# plotting
import matplotlib.pyplot as plt
import pyvista

# dolfinx and meshing
import numpy as np
from dolfinx import geometry
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
# from dolfinx.io import VTXWriter, gmshio, XDMFFile
from dolfinx.mesh import create_unit_square,create_rectangle, CellType, meshtags, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,Measure,
                 div, dot, grad, ds, dx, inner, lhs, nabla_grad, rhs, sym,
                 SpatialCoordinate, conditional, ge, le)

from dx_utils import create_obst, write_x_parview, store_array, init_db, write_values_to_json
import time

"""
    if comm.rank == 0:
        gathered_array = np.empty(mass_flow.size, dtype=np.float64)
        gathered_array2 = np.empty(mass_flow.size, dtype=np.float64)
    else:
        gathered_array, gathered_array2 = None, None
"""

def mfl_press(comm,x_max, mesh, mesh_tag, u_n, p):
    # Extract the normal component of velocity (u_x in 2D)
    u_sub = u_n.sub(0)
    # Define measures and spatial coordinates
    dx = Measure("dx", domain=mesh) #, subdomain_data=mesh_tag)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    tol = 5e-2
    mfl, mass_flow, p_loc, pressure_avg = np.array([]), None, None, np.array([])
    #if dx.subdomain_data() != 0:
    for i in np.array([0+2*tol, x_max/2, x_max-2*tol]):
        slice_condition = conditional(ge(x[0], i-tol), 1.0, 0.0) * conditional(le(x[0], i+tol), 1.0, 0.0)
        # Calculate mass flow rate at the current slice
        mass_flow_local = assemble_scalar(form(u_sub *slice_condition* dx))
        mass_flow = mesh.comm.allreduce(mass_flow_local, op=MPI.SUM)
        mfl = np.append(mfl, mass_flow)
        # Calculate average pressure at the current slice
        pressure_loc = assemble_scalar(form(p *slice_condition* dx))
        p_loc = mesh.comm.allreduce(pressure_loc, op=MPI.SUM)
        pressure_avg = np.append(pressure_avg, p_loc)
    #print(mass_flow)
    if mesh.comm.rank==0:
        print("mass_flow: ",mfl) # , "pressure: ", pressure_avg)
    return mfl, pressure_avg

def plot_para_velo(ax, mesh, u_n, p_n, t, length, pres, Ox, r, tol):
    # if mesh.comm.rank == 0:
    rank = comm.Get_rank()
    size = comm.Get_size()
    y = np.linspace(0+tol, length, int(length/tol))
    points = np.zeros((3, int(length/tol)))
    points[1] = y
    points[0] = 0
    
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    res_loc, res_loc1, res_loc2 = [[],[],[],None], [[],[],[], None], [[],[],[], None]
    res0, res1, res2 = [],[],[]
    #cells, cells1, cells2, p_o_p, p_o_p1, p_o_p2 = [], [], [], [], [], []
    
    def get_points_of_cells(bb_tree, msh, point): #, pop, cell):
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.T)
        colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, point.T)
        # Choose one of the cells that contains the point

        if colliding_cells != None:
            pop, cell =  [],[]
            for i, point in enumerate(points.T):
                if len(colliding_cells.links(i)) > 0:
                    pop.append(point)
                    cell.append(colliding_cells.links(i)[0])
            if pop != []:
                pop = np.array(pop, dtype=np.float64)
                u_val = u_n.eval(pop, cell)
                p_val = p_n.eval(pop, cell)
                return [pop, u_val, p_val, rank]
        return [None, None, None, None]

    def gather_and_sort(pop, u_val, p_val):
        # Gather data from all ranks
        all_pop = mesh.comm.gather(pop, root=0)
        all_u_val = mesh.comm.gather(u_val, root=0)
        all_p_val = mesh.comm.gather(p_val, root=0)
        
        if rank == 0:
            comb_pop = [arr for arr in all_pop if arr is not None]
            comb_u_val = [arr for arr in all_u_val if arr is not None]
            comb_p_val = [arr for arr in all_p_val if arr is not None]
            # Combine gathered data
            combined_pop = np.concatenate(comb_pop)
            combined_u_val = np.concatenate(comb_u_val)
            combined_p_val = np.concatenate(comb_p_val)
    
            # Create a sorting index based on u_val[:,0]
            sort_index = np.argsort(combined_u_val[:, 0])
            # Sort all arrays using this index
            sorted_pop = combined_pop[sort_index]
            sorted_u_val = combined_u_val[sort_index]
            sorted_p_val = combined_p_val[sort_index]
    
            return sorted_pop, sorted_u_val, sorted_p_val
        else:
            return None, None, None
    
    # get velocity procile values at x[0] = 0
    res_loc = get_points_of_cells(bb_tree, mesh, points) # , p_o_p, cells)
    p_o_p, u_values,_ = gather_and_sort(res_loc[0], res_loc[1], res_loc[2])
    
    # get velocity procile values at x of obstacle
    y2 = np.linspace(0+tol, length-(r+tol), int(length/tol))
    points[1], points[0] = y, Ox
    res_loc1 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p1, cells1)
    p_o_p1, u_values1,_ = gather_and_sort(res_loc1[0], res_loc1[1], res_loc1[2])
    
    # get velocity profile at end of canal
    points[1], points[0] = y, length
    res_loc2 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p2, cells2)s
    p_o_p2, u_values2,_ = gather_and_sort(res_loc2[0], res_loc2[1], res_loc2[2])
    
    #for i in [res_loc,res_loc1,res_loc2]:
    #    if i[0] is not None:
    #        print(f"not empty rank is {int(i[3]):d}\n",i[0][:,1], i[1][:,0])

    #print("val: ", res1, f" rank is {res0[3]}") #, " val1: ", len(res0[0][:])) #, " val2: ", res[2])
    if ax is not None:
        ax.set_title("Velocity over x-Axis")
        ax.plot(p_o_p[:, 1], u_values[:,0], "k", linewidth=2, label="x=0")
        ax.plot(p_o_p1[:, 1], u_values1[:,0], "y", linewidth=2, label=r"x=%s"%(Ox))
        ax.plot(p_o_p2[:, 1], u_values2[:,0], "b", linewidth=2, label=r"x=%s"%(length))
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("Velocity u")
        # If run in parallel as a python file, we save a plot per processor
        plt.savefig(f"para_plot/u_n_p_{int(r):d}_{int(pres):d}_{int(t*100):d}.pdf") #25_{int(pres):d}_{int(t*100):d}.pdf")
    if rank == 0:
        return p_o_p[:, 1], u_values[:,0], p_o_p1[:, 1], u_values1[:,0], p_o_p2[:,1], u_values2[:,0]
    else:
        return None, None, None, None, None, None
    #return res0[0][1][:], res0[1][0][:], res1[0][1][:], res1[1][0][:],res2[0][1][:], res2[1][0][:]

def plot_2dmesh(V, mesh, u_n, c):
    topology, cell_types, geo = vtk_mesh(V)
    values = np.zeros((geo.shape[0], 3), dtype=np.float64)
    values[:, :len(u_n)] = u_n.x.array.real.reshape((geo.shape[0], len(u_n)))
    
    # Create a point cloud of glyphs
    function_grid = pyvista.UnstructuredGrid(topology, cell_types, geo)
    function_grid["u"] = values
    glyphs = function_grid.glyph(orient="u", factor=0.2)
    
    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
    
    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
        plotter.screenshot(f"canal_{c:d}.png")
    else:
        #f"para_plot/u_n_p_canal_test")#{int(pres):d}
        fig_as_array = plotter.screenshot(f"glyphs_{c:d}.png")


def run_sim(comm, height=1, length=3,pres=8,T=.5,num_steps=500,r=0, file=False, run=1, tol=.07):
    Ox = length/2
    if run==0:
        mesh = create_rectangle(comm,[[0,0], [length, height]],[int(length*25),int(height*25)])
    if run == 1:
        mesh = create_unit_square(comm, 100, 100)
    if run == 2:
        # create_obst(comm,H=1, L=3,r=.3, Ox=1.5, lc=.07):
        mesh, ct, ft, inlet_marker,outlet_marker, upper_wall_marker, lower_wall_marker = create_obst(comm,height, length, r, Ox, tol)
    
    debug = False
    t = 0
    pres = pres * length
    dt = T / num_steps

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    fdim = mesh.topology.dim - 1

    #write_x_parview(mesh,ct,ft, "my_mesh")

    wall_dofs = locate_dofs_topological(V, fdim, ft.find(upper_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip1 = dirichletbc(u_noslip, wall_dofs, V)

    wall_dofs = locate_dofs_topological(V, fdim, ft.find(lower_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip2 = dirichletbc(u_noslip, wall_dofs, V)
    
    inflow_dofs = locate_dofs_topological(Q, fdim, ft.find(inlet_marker))
    bc_inflow = dirichletbc(PETSc.ScalarType(pres), inflow_dofs, Q)
    
    outflow_dofs = locate_dofs_topological(Q, fdim, ft.find(outlet_marker))
    bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
    bcu = [bc_noslip1, bc_noslip2]
    bcp = [bc_inflow, bc_outflow]
    if debug:
        print("<< done boundary conditions >>")
    
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
    if debug:
        print("<< formulated function and solvers >>")
    if file:
        from pathlib import Path
        folder = Path("results")
        folder.mkdir(exist_ok=True, parents=True)
        vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
        vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
        vtx_u.write(t)
        vtx_p.write(t)

    def inflow(x):
        return np.isclose(x[0], 0)
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, inflow
    )

    values = np.full(boundary_facets.shape, 1, dtype=np.int32)
    facet_tag = meshtags(mesh, mesh.topology.dim - 1, boundary_facets, values)
    # add a simple plot output 
    mfl_old = 0    
   
    if comm.rank==0:
        p, pat = init_db(f"parametric_canal_{r:.2f}_{pres:.1f}", False)

        p.put_annotation("metadata", write_values_to_json([height, length, pres, T, num_steps, r, Ox, tol],
                                                         ["height", "length", "pressure_delta", "simulation_time", 
                                                           "steps", "radius", "obstacle_location_x","meshing_size/tol"]))
    
    if debug:
        print("<< starting loop >>")
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
        if (i !=0 and i!=1 and (i%200)==0): # and comm.rank == 0:
            mfl, _ = mfl_press(comm,length, mesh, facet_tag, u_n, p_n)
            ax = None
            x1, y1, x2, y2, x3, y3 = plot_para_velo(ax,mesh, u_n, p_n, T, length, pres,Ox, r, tol)
            if comm.rank==0:
                store_array(mfl, "massflowrate", pat,p,t)
                # store_array(pa, "pressure_avg", pat,p)           
                store_array(x1, "x_at_0", pat,p,t)
                store_array(y1,  "y_at_0", pat,p,t)
                store_array(x2,  "x_at_5", pat,p,t)
                store_array(y2,  "y_at_5", pat,p,t)
                store_array(x3,  "x_at_1", pat,p,t)
                store_array(y3,  "y_at_1", pat,p,t)
            #dist = np.abs(mfl[0] - mfl_old)
            #mfl_old = mfl[0]
            #print(t, "dist: ",dist)
            #if (dist < 3e-3):
            #    print("terminating")
            #    break

        if file:
            # Write solutions to fileV
            vtx_u.write(t)
            vtx_p.write(t)
    if mesh.comm.rank == 0:
        #plot_2dmesh(V, mesh, u_n, 2)
        p.freeze()
    # Close xmdf file
    if file:
        vtx_u.close()
        vtx_p.close()
    b1.destroy()
    b2.destroy()
    b3.destroy()
    solver1.destroy()
    solver2.destroy()
    solver3.destroy()
    return u_n, p_n, V, mesh

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=231,T=.8,num_steps=500,r=.75,file=False,run=2, tol=0.05)
    for r in np.linspace(0.01,0.75,74):
        for p in np.linspace(2,230, 115):
            if r < 0.0112 and p<107:
                continue
            else:
                u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=p,T=.8,num_steps=800,r=r,file=False,run=2, tol=0.04)

