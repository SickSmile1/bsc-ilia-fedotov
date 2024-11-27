from mpi4py import MPI

# saving results
import dtoolcore
import dtoolcore.utils as utils
import time
import numpy as np
import json

from dolfinx import geometry
from dolfinx.io import VTXWriter, gmshio, XDMFFile
from dolfinx.fem import assemble_scalar, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.plot import vtk_mesh
from ufl import Measure, SpatialCoordinate, conditional, ge, le
import gmsh
import pyvista

def mfl_press(comm,x_max, mesh, mesh_tag, u_n):
    # Extract the normal component of velocity (u_x in 2D)
    u_sub = u_n.sub(0)
    # Define measures and spatial coordinates
    dx = Measure("dx", domain=mesh) #, subdomain_data=mesh_tag)
    x = SpatialCoordinate(mesh)
    tol = 5e-2
    mfl, mass_flow, p_loc, pressure_avg = np.array([]), None, None, np.array([])
    for i in np.array([0+2*tol, x_max/2, x_max-2*tol]):
        slice_condition = conditional(ge(x[0], i-tol), 1.0, 0.0) * conditional(le(x[0], i+tol), 1.0, 0.0)
        # Calculate mass flow rate at the current slice
        mass_flow_local = assemble_scalar(form(u_sub *slice_condition* dx))
        mass_flow = mesh.comm.allreduce(mass_flow_local, op=MPI.SUM)
        mfl = np.append(mfl, mass_flow)
        # Calculate average pressure at the current slice
    if mesh.comm.rank==0:
        print("mass_flow: ",mfl) # , "pressure: ", pressure_avg)
    return mfl

def plot_para_velo(ax, mesh, u_n, t, length, pres, Ox, r, tol):
    rank = mesh.comm.Get_rank()
    y = np.linspace(0+tol, length, int(length/tol))
    points = np.zeros((3, int(length/tol)))
    points[1] = y
    points[0] = 0
    
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    res_loc, res_loc1, res_loc2 = [[],[],[],None], [[],[],[], None], [[],[],[], None]
    
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
                return [pop, u_val, rank]
        return [None, None, None, None]

    def gather_and_sort(pop, u_val, p_val):
        # Gather data from all ranks
        all_pop = mesh.comm.gather(pop, root=0)
        all_u_val = mesh.comm.gather(u_val, root=0)
        
        if rank == 0:
            comb_pop = [arr for arr in all_pop if arr is not None]
            comb_u_val = [arr for arr in all_u_val if arr is not None]
            # Combine gathered data
            combined_pop = np.concatenate(comb_pop)
            combined_u_val = np.concatenate(comb_u_val)
    
            # Create a sorting index based on u_val[:,0]
            sort_index = np.argsort(combined_u_val[:, 0])
            # Sort all arrays using this index
            sorted_pop = combined_pop[sort_index]
            sorted_u_val = combined_u_val[sort_index]
    
            return sorted_pop, sorted_u_val
        else:
            return None, None, None
    
    # get velocity procile values at x[0] = 0
    res_loc = get_points_of_cells(bb_tree, mesh, points) # , p_o_p, cells)
    p_o_p, u_values = gather_and_sort(res_loc[0], res_loc[1], res_loc[2])
    
    # get velocity procile values at x of obstacle
    y2 = np.linspace(0+tol, length-(r+tol), int(length/tol))
    points[1], points[0] = y, Ox
    res_loc1 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p1, cells1)
    p_o_p1, u_values1 = gather_and_sort(res_loc1[0], res_loc1[1], res_loc1[2])
    
    # get velocity profile at end of canal
    points[1], points[0] = y, length
    res_loc2 = get_points_of_cells(bb_tree, mesh, points) #, p_o_p2, cells2)s
    p_o_p2, u_values2 = gather_and_sort(res_loc2[0], res_loc2[1], res_loc2[2])
    
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
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    #if not pyvista.OFF_SCREEN:
    #    plotter.show()
    #    plotter.screenshot(f"canal_{c:d}.png")
    #f"para_plot/u_n_p_canal_test")#{int(pres):d}
    rank = mesh.comm.rank
    fig_as_array = plotter.screenshot(f"velocity_graph_{c:.2f}_{rank:d}.png")
    plotter.close()

def create_obst(comm,H=.41, L=2.2,r=.3, Ox=1.5, lc=.07):
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
        
        factory.addCircleArc(5, 6, 7, 13)
        
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

        # gmsh.option.setNumber("Mesh.Algorithm", 8)
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write("my_mesh.msh")

    infl = comm.bcast(infl, root=0)
    outfl = comm.bcast(outfl, root=0)
    upper = comm.bcast(upper, root=0)
    lower = comm.bcast(lower, root=0)
    gmsh.model = comm.bcast(gmsh.model, root=0)
    mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
    return mesh, ct, ft, infl, outfl, upper, lower

def write_x_parview(msh,ct,ft, name):
    with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}_ct']/Geometry")
        file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}_ft']/Geometry")

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
    print(f"Dataset '{name:s}' saved at {fpath:s}")

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
