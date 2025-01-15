from mpi4py import MPI

# dolfinx and meshing
import numpy as np
import matplotlib.pyplot as plt
from dx_sim_iter import run_sim
from dx_utils import create_obst

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters or in a loop.")
    parser.add_argument("--single", type=int, help="single or loop run")
    parser.add_argument("--save", type=int, help="Save data to dtool db")
    parser.add_argument("--tol", type=float, help="Mesh raster size")
    parser.add_argument("--info", action="store_true", help="Show information about parameters without running the simulation")
    return parser.parse_args()

def run_iterative(comm, save, tol):
    """
    Run simulations for a range of radii and pressures.

    This function iterates over the given ranges of radii and pressures,
    running a simulation for each combination.

    :param comm:
        MPI communicator object
    :type comm:
        mpi4py.MPI.comm
    :param radius_range:
        Range of radii to simulate
    :type radius_range:
        numpy.ndarray
    :param pressure_range:
        Range of pressures to simulate
    :type pressure_range:
        numpy.ndarray
    """
    for pres in np.linspace(55, 65, 10):
        min_pres = 49
        pg_pres = 100
        p_old, pop = run_sim(comm, height=1, length=10,pres=pres,T=.8,num_steps=1000, save=1, tol=.02, 
                             mesh_created=False, meshed=None, new_membrane=True)
        while pg_pres > min_pres:      
            p_old, pop = run_sim(comm, height=1, length=10,pres=pres,T=.8,num_steps=1000, save=1, tol=.02, 
                 mesh_created=False, meshed=None, new_membrane=False, p_old=p_old, pg=(pg_pres*10) )
            pg_pres -= 5

def run_single_sim(comm,pres,p_old,pg_pres):
    p_old, pop = run_sim(comm, height=1, length=10,pres=pres,T=.8,num_steps=1000, save=1, tol=.02, 
                 mesh_created=False, meshed=None, new_membrane=False, p_old=p_old, pg=(pg_pres*10))

def show_parameter_info():
    print("## Parameter Information")
    print("This script accepts the following parameters for single run:")
    print("  --radius: The radius for the simulation (float)")
    print("  --length: The length for the simulation (float)")
    print("  --pressure: The pressure for the simulation (float)")
    print("\nExample usage:")
    print("  mpirun -np 4 python3 sim_mpi_iterative.py --save 0 --single 1 --tol 0.03")
    print("This script accepts the following parameters for loop run:")
    print("  --length: The length for the simulation (float)")
    print("  --radius_range: The radius for the simulation (start,end,step)")
    print("  --pressure_range: The pressure for the simulation (start,end,step)")    
    print("\nExample usage:")
    print("  mpirun -np 4 python3 sim_mpi_iterative.py --save 0 --single 1 --tol 0.03")

if __name__ == '__main__':
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    if args.save == 1:
        save = True
    else:
        save = False
    if args.tol is None:
        tol = 0.3
    else:
        tol = args.tol

    if args.single == 1:
        run_single_sim(comm, 57.5,[374.60,201.749],50)
    elif args.single == 0:
        run_iterative(comm, save, tol)


