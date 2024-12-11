from dolfinx import geometry
from petsc4py import PETSc
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import create_unit_square,create_rectangle, CellType, meshtags, locate_entities_boundary
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
             div, dot, grad, ds, dx, inner, lhs, nabla_grad, rhs, sym,
             SpatialCoordinate, conditional)

from dolfinx.io import VTXWriter
import numpy as np

from dx_utils import (create_obst, gather_and_sort, get_pop_cells, write_x_parview, store_array, init_db,
                    write_values_to_json, mfl_press, get_unsorted_arrays,get_pop_cells, pops_cells, 
                      plot_2dmesh, zetta, update_membrane_mesh)

import matplotlib.pyplot as plt


def run_sim(comm, height=1, length=10,pres=20,T=.8,num_steps=1000, save=1, tol=.03, mesh_created=False, meshed=None, new_membrane=False, p_old=None, pg=0):
    # set obstacle location to center
    Ox = length/2
    # disable saving to .bp file
    file = True
    # breaking condition for mpi
    break_flag = False
    """if run==0:
        # this was the initial run to see reference values
        mesh = create_rectangle(comm,[[0,0], [length, height]],[int(length*25),int(height*25)])
    if run == 1:
        # this option is the dolfinx intendet way of creating a mesh,
        # problems may arise with boundary conditions, if set from "locate_dofs_topological" 
        # as problems remained unresolved run==2 was created
        mesh = create_unit_square(comm, 100, 100)"""
    mesh, vtx_u, vtx_p = None, None, None
    # manually create mesh
    if new_membrane:
        mesh, ct, ft, inlet_marker,outlet_marker, upper_wall_marker, lower_wall_marker = update_membrane_mesh(comm,height, length, first=True)
    elif not new_membrane: #comm,H, L, lc,p0, pl, pg, first=False
        mesh, ct, ft, inlet_marker,outlet_marker, upper_wall_marker, lower_wall_marker = update_membrane_mesh(comm,height, length, tol, np.max(p_old), np.min(p_old), pg, first=False)
    else:
        print("no mesh provided")
        return 0
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
        folder = Path(f"results_{pres}/")
        folder.mkdir(exist_ok=True, parents=True)
        vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
        vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
        vtx_u.write(t)
        vtx_p.write(t)

    # add a simple plot output

    # this value can be used to break the run if the massflowrate change falls below 3e-3 in the loop
    mfl_old, mfl = 0,0

    relative_tolerance = 1.e-4
   
    if mesh.comm.rank==0 and save:
        # initialize dtool dataset
        if new_membrane == True:
            p, pat = init_db(f"iterative_canal_{pres:.1f}", False)
        else:
            pr = p_old[-1]
            p, pat = init_db(f"iterative_canal_{float(pr):.1f}_{pres:.1f}", False)
        p.put_annotation("metadata", write_values_to_json([height, length, pres, T, num_steps, Ox, tol],
                                                         ["height", "length", "pressure_delta", "simulation_time", 
                                                           "steps", "obstacle_location_x","meshing_size/tol"]))

    
    x = np.linspace(length/2-1, length/2+1, 500)
    if new_membrane==True:
        y = np.ones(500)-.05
    else:
        y = zetta(np.max(p_old), np.min(p_old), pg, num=500)
        y -= .05
    r = np.vstack((x, y, np.zeros(x.size)))
    center_height = np.min(y)
    # calculate pressure at membrane
    r_p = np.array(r, dtype=np.float64)
    press_pop, cell_press = pops_cells(r_p, mesh)
    #print(press_pop, r)
    press_p_o_p = np.array(press_pop, dtype=np.float64)
    # calculate pressure/velocity at different cross-sections
    pop, cell = get_pop_cells(height, length/2-1, mesh)
    pop_center, cell_center = get_pop_cells(center_height-0.05, Ox, mesh)
    pop_end, cell_end = get_pop_cells(height, length/2+1, mesh)
    p_o_p, p_o_p_center, p_o_p_end = np.array(pop, dtype=np.float64),np.array(pop_center, dtype=np.float64),np.array(pop_end, dtype=np.float64)
    pp = 0
    # plot_2dmesh(V, mesh, u_n, pg)
    if debug:
        print("<< starting loop >>")
    for i in range(num_steps):
        # code from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
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
        
        if i%50==0 and file:
            vtx_u.write(t)
            vtx_p.write(t)
        # write data to dataset
        if (i !=0 and i!=1 and (i%200)==0): # and comm.rank == 0:
            mfl1, _ = mfl_press(mesh.comm,length, mesh, None, u_n, p_n)
            flux, mfl = None, None
            # print("\n<<--    this is pn     -->>",p_n.x.array[:])
            pop_p, yp, pp = get_unsorted_arrays(press_p_o_p, cell_press, u_n, p_n)
            pop, y1, p1 = get_unsorted_arrays(p_o_p, cell, u_n, p_n)
            pop1, y2, p2 = get_unsorted_arrays(p_o_p_center, cell_center, u_n, p_n)
            pop2, y3, p3 = get_unsorted_arrays(p_o_p_end, cell_end, u_n, p_n)
            pop_p, yp, pp = gather_and_sort(pop_p, yp, pp, mesh)
            pop, y1, p1 = gather_and_sort(pop, y1, p1, mesh)
            pop1, y2, p2 = gather_and_sort(pop1, y2, p2, mesh)
            pop2, y3, p3 = gather_and_sort(pop2, y3, p3, mesh)
            if mesh.comm.rank == 0:
                y_grid = np.linspace(0,height,y1.shape[0])
                y_grid2 = np.linspace(0,center_height,y2.shape[0])
                flux = np.array([np.trapz(y=y1[:,0],x=y_grid),
                       np.trapz(y=y2[:,0],x=y_grid2), 
                       np.trapz(y=y3[:,0],x=y_grid)])
                print("flux: ",flux, " flux_mean: ", np.mean(mfl1))
                store_array(flux, "flux_trapz", pat, p, t)
                store_array(mfl1, "massflowrate", pat,p,t)
                store_array(y1,  "y_at_0", pat,p,t)
                store_array(y2,  "y_at_5", pat,p,t)
                store_array(y3,  "y_at_1", pat,p,t)
                store_array(p1,  "p_at_0", pat,p,t)
                store_array(p2,  "p_at_5", pat,p,t)
                store_array(p3,  "p_at_1", pat,p,t)
                store_array(pp,"p_courve", pat,p, t)
                store_array(yp,"y_courve", pat,p, t)

                if mfl_old != -1 and mesh.comm.rank == 0:
                    mfl = np.mean(flux)
                    print(mfl, mfl_old)
                    mean_of_last_two = np.mean([mfl, mfl_old])
                    dm = np.abs(mfl_old - mfl)
                    relative_deviation = dm / mean_of_last_two
                    print("Absolute deviation: %g", dm)
                    print("Relative deviation: %g", relative_deviation)
                    mfl_old = mfl
                    if relative_deviation < relative_tolerance:
                        print("Relative mass flow change %g converged within relatvie tolerance %g", relative_deviation, relative_tolerance)
                        break_flag = True
            
            break_flag = mesh.comm.bcast(break_flag, root=0)
            mfl_old = mesh.comm.bcast(mfl_old, root=0)
            if break_flag or i==(num_steps-1):
                pp = mesh.comm.bcast(pp, root=0)
                pop_p = mesh.comm.bcast(pop_p, root=0)
            if break_flag:
                break
    if mesh.comm.rank == 0 and save:
        p.freeze()
        
    # plot_2dmesh(V, mesh, u_n, pg)
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
    return pp, pop_p
