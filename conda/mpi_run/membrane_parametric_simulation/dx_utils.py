from mpi4py import MPI

# saving results
import dtoolcore
import dtoolcore.utils as utils
import time
import numpy as np
# import json

from dolfinx import geometry
from dolfinx.io import VTXWriter, gmshio, XDMFFile
from dolfinx.fem import assemble_scalar, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.plot import vtk_mesh
from ufl import Measure, SpatialCoordinate, conditional, ge, le
import gmsh
import pyvista

def mfl_press(comm,x_max, mesh, mesh_tag, u_n, p):
    # Extract the normal component of velocity (u_x in 2D)
    u_sub = u_n.sub(0)
    # Define measures and spatial coordinates
    dx = Measure("dx", domain=mesh) #, subdomain_data=mesh_tag)
    x = SpatialCoordinate(mesh)
    tol = 5e-2
    mfl, mass_flow, p_loc, pressure_avg = np.array([]), None, None, np.array([])
    for i in np.array([0.5, x_max/2, x_max-.5]):
        slice_condition = conditional(ge(x[0], i-tol), 1.0, 0.0) * conditional(le(x[0], i+tol), 1.0, 0.0)
        # Calculate mass flow rate at the current slice
        mass_flow_local = assemble_scalar(form(u_sub *slice_condition* dx))
        mass_flow = mesh.comm.allreduce(mass_flow_local, op=MPI.SUM)
        mfl = np.append(mfl, mass_flow)
        # Calculate average pressure at the current slice
        pressure_loc = assemble_scalar(form(p *slice_condition* dx))
        p_loc = mesh.comm.allreduce(pressure_loc, op=MPI.SUM)
        pressure_avg = np.append(pressure_avg, p_loc)
    if mesh.comm.rank==0:
        print("mass_flow: ",mfl) # , "pressure: ", pressure_avg)
    return mfl, pressure_avg

def get_unsorted_arrays(pop, cell, u_n, p_n): #, pop, cell):
    # Find cells whose bounding-box collide with the the points
    if len(pop) > 0:
        u_val = u_n.eval(pop, cell) 
        p_val = p_n.eval(pop, cell)
        return [pop, u_val, p_val]
    else:
        return [None, None, None]

def get_pop_cells(length, x, mesh):
    y = np.linspace(0, length, 100)
    points = np.zeros((3, 100))
    points[1] = y
    points[0] = x
    pop, cell = pops_cells(points, mesh)
    return pop, cell

def pops_cells(points, mesh):
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    # Choose one of the cells that contains the point
    pop, cell = [],[]
    if colliding_cells is not None:
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                pop.append(point)
                cell.append(colliding_cells.links(i)[0])
        return pop, cell

def gather_and_sort_try(pop, u_val, p_val, mesh):
    # Gather data from all ranks
    all_pop = mesh.comm.gather(pop, root=0)
    all_u_val = mesh.comm.gather(u_val, root=0)
    all_p_val = mesh.comm.gather(p_val, root=0)
    
    if mesh.comm.rank == 0:
        if len(all_pop) < 90:
            # Filter out None values
            comb_pop = [arr for arr in all_pop if arr is not None]
            comb_u_val = [arr for arr in all_u_val if arr is not None]
            comb_p_val = [arr for arr in all_p_val if arr is not None]
            
            # Ensure all arrays have the same number of dimensions
            max_dim = max(arr.ndim for arr in comb_u_val + comb_p_val)
            comb_u_val = [np.atleast_2d(arr) if arr.ndim < max_dim else arr for arr in comb_u_val]
            comb_p_val = [np.atleast_2d(arr) if arr.ndim < max_dim else arr for arr in comb_p_val]
            
            try:
                # Combine gathered data
                combined_pop = np.concatenate(comb_pop)
                combined_u_val = np.concatenate(comb_u_val)
                combined_p_val = np.concatenate(comb_p_val)
                
                # Create a sorting index based on combined_pop[:, 1]
                sort_index = np.argsort(combined_pop[:, 1])
                
                # Sort all arrays using this index
                sorted_pop = combined_pop[sort_index]
                sorted_u_val = combined_u_val[sort_index]
                sorted_p_val = combined_p_val[sort_index]
            except ValueError as e:
                print(f"Error during concatenation or sorting: {e}")
                # If concatenation fails, return the original data
                sorted_pop, sorted_u_val, sorted_p_val = comb_pop, comb_u_val, comb_p_val
        else:
            sorted_pop = all_pop
            sorted_u_val = all_u_val
            sorted_p_val = all_p_val
    else:
        sorted_pop = sorted_u_val = sorted_p_val = None

    # Broadcast results to all processes
    pop_res = mesh.comm.bcast(sorted_pop, root=0)    
    uval_res = mesh.comm.bcast(sorted_u_val, root=0)    
    pval_res = mesh.comm.bcast(sorted_p_val, root=0)    
    
    return pop_res, uval_res, pval_res

def gather_and_sort(pop, u_val, p_val, mesh):
    # Gather data from all ranks
    pop_res, uval_res, pval_res = None, None, None
    sorted_pop, sorted_u_val, sorted_p_val = None, None, None
    all_pop = mesh.comm.gather(pop, root=0)
    all_u_val = mesh.comm.gather(u_val, root=0)
    all_p_val = mesh.comm.gather(p_val, root=0)
    if mesh.comm.rank == 0:
        if len(all_pop) < 90:
            comb_pop = [arr for arr in all_pop if arr is not None]
            comb_u_val = [arr for arr in all_u_val if arr is not None]
            comb_p_val = [arr for arr in all_p_val if arr is not None]
            # Combine gathered data
            combined_pop = np.concatenate(comb_pop)
            combined_u_val = np.concatenate(comb_u_val)
            combined_p_val = np.concatenate(comb_p_val)
    
            # Create a sorting index based on u_val[:,0]
            sort_index = np.argsort(combined_pop[:, 1])
            #print("\n0:",combined_pop[:,0],"\n1:", combined_pop[:,1],"\ncombpopsize",combined_pop.shape)
            #print("\n0 u_val:",combined_u_val[:,0],"\n1 u_val:", combined_u_val[:,1],"\size",combined_u_val.shape)
            #print("\n0:p_val",combined_p_val[:,0],"\n1:","\ncombpopsize",combined_p_val.shape)
            # Sort all arrays using this index
            sorted_pop = combined_pop[sort_index]
            sorted_u_val = combined_u_val[sort_index]
            sorted_p_val = combined_p_val[sort_index]
        else:
            sorted_pop = all_pop
            sorted_u_val = all_u_val
            sorted_p_val = all_p_val
    pop_res = mesh.comm.bcast(sorted_pop, root=0)    
    uval_res = mesh.comm.bcast(sorted_u_val, root=0)    
    pval_res = mesh.comm.bcast(sorted_p_val, root=0)    
    return pop_res, uval_res, pval_res

def plot_para_velo(ax, mesh, u_n, p_n, t, length, pres, Ox, r, tol):
    rank = mesh.comm.Get_rank()
    
    res_loc, res_loc1, res_loc2 = [[],[],[],None], [[],[],[], None], [[],[],[], None]

    p_o_p,p_o_p1,p_o_p2,u_values,u_values1,u_values2,p_values,p_values1,p_values2=None,None,None,None,None,None,None,None,None
    # get velocity procile values at x[0] = 0
    res_loc = get_points_of_cells(bb_tree, mesh, points) # , p_o_p, cells)
    p_o_p, u_values, p_values = gather_and_sort(res_loc[0], res_loc[1], res_loc[2])
    
    # get velocity procile values at x of obstacle
    res_loc1 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p1, cells1)
    p_o_p1, u_values1, p_values1 = gather_and_sort(res_loc1[0], res_loc1[1], res_loc1[2])
    
    # get velocity profile at end of canal
    res_loc2 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p2, cells2)s
    p_o_p2, u_values2, p_values2 = gather_and_sort(res_loc2[0], res_loc2[1], res_loc2[2])
    
    if rank == 0:
        return p_o_p[:, 1], u_values[:,0], p_o_p1[:, 1], u_values1[:,0], p_o_p2[:,1], u_values2[:,0], p_values[:,0],p_values1[:,0],p_values2[:,0]
    else:
        return None, None, None, None, None, None, None, None, None

def zetta(p0, pl, pg, L=2,T=30, num=100):
    """
    Calculate the zetta value for a given location in a membrane canal.

    This function computes the zetta value based on the pressures at different points
    of a membrane and the location within the canal.

    Parameters:
    -----------
    T : float
        The stiffness of the membrane
    p0 : float
        The pressure at the beginning of the membrane.
    pl : float
        The pressure at the end of the membrane.
    pg : float
        The outer pressure of the membrane.
    L : float
        The length of the membrane.
    x : float
        The location in the canal for which to calculate zetta.

    Returns:
    --------
    float
        The calculated zetta value at the given location.

    Notes:
    ------
    The function uses the following formula:
    zetta = 1/T * (1/2 * (pg - p0) * x**2 + pd/(6*L) * x**3 - 1/6 * (3*pg - 2*p0 - pl)* L* x)
    where pd = p0 - pl

    The constant T is not defined in the function and should be provided or defined elsewhere.

    Example:
    --------
    >>> zetta(100, 80, 120, 10, 5)
    # Returns the zetta value at the midpoint of a 10-unit long membrane
    """
    x = np.linspace(0,2,num)
    print("p0: ",p0, " pg: ",pg, " pl: ",pl)
    assert (p0<pg)
    pd = p0-pl
    res = 1/T * (1/2 * (pg - p0) * x**2 + pd/(6*L) * x**3 - 1/6 * (3*pg - 2*p0 - pl)* L * x )
    res += 1
    assert np.min(res) > 0
    return res

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
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    #if not pyvista.OFF_SCREEN:
    #    plotter.show()
    #    plotter.screenshot(f"canal_{c:d}.png")
    #f"para_plot/u_n_p_canal_test")#{int(pres):d}
    rank = mesh.comm.rank
    fig_as_array = plotter.screenshot(f"velocity_graph_{int(c):d}.png")
    plotter.close()

def create_obst(comm,H=1, L=3,r=.3, Ox=1.5, lc=.07):
    """
    Create a 2D mesh of a canal with a circular obstacle using GMSH.

    This function generates a mesh for a rectangular canal with a circular obstacle
    on the upper wall. It uses GMSH for mesh generation and MPI for parallel processing.

    :param comm: MPI communicator
        The MPI communicator object for parallel processing.
    :param H: float, optional
        Height of the canal (default is 1).
    :param L: float, optional
        Length of the canal (default is 3).
    :param r: float, optional
        Radius of the circular obstacle (default is 0.3).
    :param Ox: float, optional
        X-coordinate of the obstacle center (default is 1.5).
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
        r = r * cm
        Lc1 = lc
        
        # We start by defining some points and some lines. To make the code shorter we
        # can redefine a namespace:
        factory = gmsh.model.geo
        model = gmsh.model
        
        factory.addPoint(0, 0, 0, Lc1, 1)
        factory.addPoint(l1, 0, 0, Lc1, 2)
        factory.addPoint(l1, h1 , 0, Lc1, 3)
        factory.addPoint(0, h1, 0, Lc1, 4)
        
        factory.addPoint(Ox-r, h1, 0, Lc1, 5)
        factory.addPoint(Ox, h1, 0, Lc1, 6)
        factory.addPoint(Ox+r, h1, 0, Lc1, 7)
        
        factory.addLine(1, 2, 8)
        factory.addLine(2, 3, 9)
        factory.addLine(3, 7, 10)
        factory.addLine(5, 4, 11)
        factory.addLine(4, 1, 12)
        
        factory.addCircleArc(7, 6, 5, 13)
        
        # Define the inner curve loop (the circle arc)
        factory.addCurveLoop([-13], 14)
        
        # Define the outer curve loop
        factory.addCurveLoop([8, 9, 10, -13, 11, 12], 15)
        
        # Create the plane surface with a hole
        factory.addPlaneSurface([15], 16)
        factory.synchronize()
        
        upper = model.addPhysicalGroup(1, [10, -13, 11])
        model.setPhysicalName(1, upper, "upper_wall")
        outfl = model.addPhysicalGroup(1, [9])
        model.setPhysicalName(1, outfl, "outflow")
        infl = model.addPhysicalGroup(1, [12])
        model.setPhysicalName(1, infl, "inflow")
        lower = model.addPhysicalGroup(1, [8])
        model.setPhysicalName(1, lower, "lower_wall")

        gmsh.model.addPhysicalGroup(2, [16], tag=5, name="Domain")
        factory.synchronize()
        #gmsh.option.setNumber("Mesh.ElementOrder", 1)
        #gmsh.option.setNumber("Mesh.RecombineAll", 0)
        #gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
    infl = comm.bcast(infl, root=0)
    outfl = comm.bcast(outfl, root=0)
    upper = comm.bcast(upper, root=0)
    lower = comm.bcast(lower, root=0)
    gmsh.model = comm.bcast(gmsh.model, root=0)
    #gmsh.write(f"mesh_r{r:.1f}.msh")
    mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
    return gmsh.mode, mesh, ct, ft, infl, outfl, upper, lower

def update_membrane_mesh(comm,H, L, lc=.03, p0=0, pl=0, pg=0, first=False):
    #comm,H=1, L=3,r=.3, Ox=1.5, lc=.07
    def define_membrane(factory, begin, end, l1, lc1,L):
        memb = zetta(p0, pl, pg,2,1000)
        startpoint = (L/2)-(L/10)
        endpoint = (L/2)+(L/10)
        x = np.linspace(startpoint, endpoint, 100)
        lines = []
        points = []
        points.append(begin)
        for i in range(len(x)-2):
            new_point = factory.addPoint(x[i+1], memb[i+1], 0, lc1)
            points.append(new_point)
            lines.append(factory.addLine(points[-2], points[-1]))
        lines.append(factory.addLine(points[-1],end))
        return lines, points
    
    silent = False
    model_rank = 0
    infl, outfl, upper, lower = [],[],[],[]
    gmsh.initialize()
    if silent:
        gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("canal")
    #gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)
    
    cm = 1 # e-02 # not needed for our sim
    h1 = H * cm if H is not None else 1
    l1 = L * cm if L is not None else 10
    Lc1 = lc
    
    # We start by defining some points and some lines. To make the code shorter we
    # can redefine a namespace:
    factory = gmsh.model.occ
    model = gmsh.model
    if comm.rank==0:
        lowerleft = factory.addPoint(0, 0, 0, Lc1)
        lowerright = factory.addPoint(l1, 0, 0, Lc1)
        upperright = factory.addPoint(l1, h1 , 0, Lc1)
        upperleft = factory.addPoint(0, h1, 0, Lc1)
        
        begin = factory.addPoint(L/2-L/10, h1, 0, Lc1)
        end = factory.addPoint(L/2+L/10, h1, 0, Lc1)
        
        inflow_line = factory.addLine(lowerleft, upperleft)
        upper_wall_left = factory.addLine(upperleft, begin)
        upper_wall_right = factory.addLine(end, upperright)
        outflow_line = factory.addLine(upperright, lowerright)
        lower_wall = factory.addLine(lowerright, lowerleft)
        lines = None
        if first:
            lines = [factory.addLine(begin, end)]
        else:
            # add obstacle form
            lines, points = define_membrane(factory, begin, end, l1, Lc1, L)

        # Define the outer curve loop
        o_loop = factory.addCurveLoop([inflow_line, upper_wall_left, *lines,
                                      upper_wall_right, outflow_line, lower_wall])
        
        # Create the plane surface with a hole
        surface = factory.addPlaneSurface([o_loop])
        factory.synchronize()
        upper = model.addPhysicalGroup(dim=1, tags=[upper_wall_left, *lines, upper_wall_right],tag=1,name="upper_wall")
        outfl = model.addPhysicalGroup(dim=1, tags=[outflow_line], tag=2, name="outflow")
        infl = model.addPhysicalGroup(dim=1, tags=[inflow_line], tag=3, name="inflow")
        lower = model.addPhysicalGroup(dim=1, tags=[lower_wall], tag=4, name="lower_wall")
        
        gmsh.model.addPhysicalGroup(dim=2, tags=[surface], tag=5, name="Domain")
        factory.synchronize()
        #gmsh.option.setNumber("Mesh.ElementOrder", 1)
        #gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.model.mesh.generate(2)
        #gmsh.write(f"mesh_{pg:.1f}.msh")
    infl = comm.bcast(infl, root=0)
    outfl = comm.bcast(outfl, root=0)
    upper = comm.bcast(upper, root=0)
    lower = comm.bcast(lower, root=0)
    gmsh.model = comm.bcast(gmsh.model, root=0)
    mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
    return gmsh, mesh, ct, ft, infl, outfl, upper, lower

def write_x_parview(msh,ct,ft, name):
    with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}_ct']/Geometry")
        file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}_ft']/Geometry")

def store_gmsh(model, name, path, p, db="dtool_db"):
    if model is not None:
        fpath = path+"/data/"
        utils.mkdir_parents(fpath)
        model.write(fpath+name+".msh")
    else:
        print("DEBUG: dx_utils/store_gmsh()\nModel is None!")
    # print(f"Dataset '{name:s}' saved at {fpath:s}")

def store_array(arr, name, path, p, t, db="dtool_db"):
    """
    Store a numpy array as a text file in a specified directory.

    This function saves a numpy array to a text file and creates the necessary
    directory structure if it doesn't exist. It also provides an option to
    include a time parameter in the file path.

    :param arr: numpy.ndarray
        The array to be stored.
    :param name: str
        The name of the dataset (used for the filename).
    :param path: str
        The base path where the data will be stored.
    :param p: object
        An object containing additional parameters (not used in this function).
    :param db: str, optional
        The name of the database (default is "dtool_db", not used in this function).
    :param t: float, optional
        A time parameter. If not -1, it will be included in the file path (default is -1).

    :return: None

    :raises: None explicitly, but may raise exceptions from called functions.

    :example:
    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> store_array(arr, "test_data", "/path/to/storage", None)
    Dataset 'test_data' saved at /path/to/storage/data/

    .. note::
        - The function uses `utils.mkdir_parents()` to create directories.
        - The array is saved using `numpy.savetxt()` with 10 decimal places precision.
        - If the input array is empty, nothing will be saved.

    .. warning::
        Ensure that the `utils` module with the `mkdir_parents` function is imported
        and available in the same environment.
    """
    if arr.size > 0:
        fpath = path+"/data/"
        fpath += f"{name:s}_{t:.2f}/"
        utils.mkdir_parents(fpath)
        np.savetxt(fpath+name+".txt", arr, fmt='%.10e')
    else:
        print("DEBUG: dx_utils/store_array()\nArray size 0 or None!")
    # print(f"Dataset '{name:s}' saved at {fpath:s}")

def init_db(dataset_name, identifier=True, db="dtool_db"):
    """
    Initialize a new dtool dataset.

    This function creates a new proto dataset using dtoolcore. It generates a unique
    database path based on the current epoch time and the provided dataset name.

    Args:
        dataset_name (str): The name of the dataset to be created.
        db (str, optional): The base directory for the database. Defaults to "dtool_db".

    Returns:
        tuple: A tuple containing two elements:
            - proto (dtoolcore.ProtoDataSet): The created proto dataset object.
            - db_path (str): The full path to the created database.

    Example:
        >>> proto, db_path = init_db("my_dataset")
        >>> print(db_path)
        dtool_db/my_dataset_1636456789.123456
    """
    epoch = time.time()
    db_path = db+"/"+dataset_name+"_"+str(epoch)
    if not identifier:
        db_path = db
    utils.mkdir_parents(db_path+"/")
    proto = dtoolcore.create_proto_dataset(dataset_name,
                db_path,
                readme_content=f"canal_data_{dataset_name:s}",
                creator_username="Ilia Fedotov")
    return proto, db_path+"/"+dataset_name

def write_values_to_json(values, names):
    """
    Write a list of values and corresponding names into a JSON variable.

    Args:
        values (list): A list of values to be written (should be numbers).
        names (list): A list of names corresponding to the values.

    Returns:
        data (json): A json list for dtool annotations

    Example:
        >>> write_values_to_json([1.5, 2.7, 3.2], ['a', 'b', 'c'])
        # This will create a file 'output.json' with the content:
        # [{"a": 1.5, "b": 2.7, "c": 3.2}]
    """
    # Check if the lengths of values and names match
    if len(values) != len(names):
        raise ValueError("The number of values must match the number of names")

    # Check if all values are numbers
    if not all(isinstance(v, (int, float)) for v in values):
        raise TypeError("All values must be numbers (integers or floats)")

    # Create the data structure
    data = [{name: value for name, value in zip(names, values)}]
    return data
