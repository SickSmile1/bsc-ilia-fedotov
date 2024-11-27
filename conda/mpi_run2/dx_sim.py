
from petsc4py import PETSc
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import create_unit_square,create_rectangle, CellType, meshtags, locate_entities_boundary
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, grad, ds, dx, inner, lhs, nabla_grad, rhs, sym,
                 SpatialCoordinate, conditional)
import tqdm.autonotebook
import numpy as np

from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)

from dx_utils import (create_obst, write_x_parview, store_array, init_db,
                    write_values_to_json, mfl_press, plot_para_velo, plot_2dmesh)

def run_sim(comm, height=.41, length=2.2,pres=8,T=8,num_steps=500,r=0, save=False, tol=.05):
    # set obstacle location to center
    Ox = length/2
    # disable saving to .bp file
    file =  False
    """if run==0:
        # this was the initial run to see reference values
        mesh = create_rectangle(comm,[[0,0], [length, height]],[int(length*25),int(height*25)])
    if run == 1:
        # this option is the dolfinx intendet way of creating a mesh,
        # problems may arise with boundary conditions, if set from "locate_dofs_topological" 
        # as problems remained unresolved run==2 was created
        mesh = create_unit_square(comm, 100, 100)"""
    
    # manually create mesh
    mesh, ct, ft, inlet_marker,outlet_marker, upper_wall_marker, lower_wall_marker = create_obst(comm,height, length, r, Ox, tol)
    
    debug = False
    t = 0
    dt = 1 / 1600                 # Time step size
    num_steps = int(T / dt)
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))     # Density
    gdim = 2


    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    
    fdim = mesh.topology.dim - 1

    class InletVelocity():
        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 4 * pres * np.sin(1 * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
            return values

    wall_dofs = locate_dofs_topological(V, fdim, ft.find(upper_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip1 = dirichletbc(u_noslip, wall_dofs, V)

    wall_dofs = locate_dofs_topological(V, fdim, ft.find(lower_wall_marker))
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc_noslip2 = dirichletbc(u_noslip, wall_dofs, V)
    
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t)
    u_inlet.interpolate(inlet_velocity)
    
    # inflow_dofs = locate_dofs_topological(V, fdim, ft.find(inlet_marker))
    bc_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
    # bc_inflow = dirichletbc(u_inlet, inflow_dofs)
    
    outflow_dofs = locate_dofs_topological(Q, fdim, ft.find(outlet_marker))
    bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
    bcu = [bc_noslip1, bc_noslip2,bc_inflow]
    bcp = [bc_outflow]
    if debug:
        print("<< done boundary conditions >>")
    
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_s = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q)
    p_.name = "p"
    phi = Function(Q)

    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)

    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    # Solver for step 1
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
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
        vtx_u = VTXWriter(mesh.comm, "dfg2D-3-u.bp", [u_], engine="BP4")
        vtx_p = VTXWriter(mesh.comm, "dfg2D-3-p.bp", [p_], engine="BP4")
        vtx_u.write(t)
        vtx_p.write(t)

    def inflow(x):
        return np.isclose(x[0], 0)
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, inflow
    )

    values = np.full(boundary_facets.shape, 1, dtype=np.int32)
    facet_tag = meshtags(mesh, mesh.topology.dim - 1, boundary_facets, values)

    # this value can be used to break the run if the massflowrate change falls below 3e-3 in the loop
    mfl_old = 0    
    
    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[length/3, height/2, 0], [length*2/3, height/2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
    if mesh.comm.rank == 0:
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

    if comm.rank==0 and save:
        # initialize dtool dataset
        p, pat = init_db(f"parametric2_canal_{r:.2f}_{pres:.1f}", False)
        p.put_annotation("metadata", write_values_to_json([height, length, pres, T, num_steps, r, Ox, tol],
                                                         ["height", "length", "pressure_delta", "simulation_time", 
                                                           "steps", "radius", "obstacle_location_x","meshing_size/tol"]))
    
    if debug:
        print("<< starting loop >>")

    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
    for i in range(num_steps):
        # code from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
        # Update current time step
        progress.update(1)
        t += dt
        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Update variable with solution form this time step
        with u_.x.petsc_vec.localForm() as loc_, u_n.x.petsc_vec.localForm() as loc_n, u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        # write data to dataset
        if (i !=0 and i!=1 and (i%1000)==0): # and comm.rank == 0:
            mfl = mfl_press(comm,length, mesh, facet_tag, u_n)
            ax = None
            x1, y1, x2, y2, x3, y3 = plot_para_velo(ax,mesh, u_, T, length, pres,Ox, r, tol)
            plot_2dmesh(V, mesh, u_, t)
            
            p_front = None
            if len(front_cells) > 0:
                p_front = p_.eval(points[0], front_cells[:1])
            p_front = mesh.comm.gather(p_front, root=0)
            p_back = None
            if len(back_cells) > 0:
                p_back = p_.eval(points[1], back_cells[:1])
            p_back = mesh.comm.gather(p_back, root=0)

            if comm.rank==0 and save:
                plot_2dmesh(V, mesh, u_n, t)
                store_array(mfl, "massflowrate", pat,p,t)
                # store_array(pa, "pressure_avg", pat,p)           
                store_array(x1, "x_at_0", pat,p,t)
                store_array(y1,  "y_at_0", pat,p,t)
                store_array(x2,  "x_at_5", pat,p,t)
                store_array(y2,  "y_at_5", pat,p,t)
                store_array(x3,  "x_at_1", pat,p,t)
                store_array(y3,  "y_at_1", pat,p,t)

                # Choose first pressure that is found from the different processors
                for pressure in p_front:
                    if pressure is not None:
                        p_diff[i] = pressure[0]
                        break
                for pressure in p_back:
                    if pressure is not None:
                        p_diff[i] -= pressure[0]
                        break
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
    if mesh.comm.rank == 0 and save:
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
    return u_n, u_n, V, mesh
