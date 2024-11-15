from mpi4py import MPI

# saving results
import dtoolcore
import dtoolcore.utils as utils
import time
import numpy as np
import json


from dolfinx.io import VTXWriter, gmshio, XDMFFile
import gmsh

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
        gmsh.model.mesh.generate(2)
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

def store_array(arr, name, path, p, db="dtool_db", t=-1):
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
        if t!= -1:
            fpath = path+f"/data/{name:s}_{t:.2f}_/"
        utils.mkdir_parents(fpath)
        np.savetxt(fpath+name+".txt", arr, fmt='%.10e')
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
    print(f"Dataset '{db_path:s}' lala")
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