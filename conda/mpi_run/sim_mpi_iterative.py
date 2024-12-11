from mpi4py import MPI

# dolfinx and meshing
import numpy as np
import matplotlib.pyplot as plt
from dx_sim_iter import run_sim
from dx_utils import create_obst

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters or in a loop.")
    parser.add_argument("--save", type=int, help="Save data to dtool db")
    parser.add_argument("--tol", type=float, help="Mesh raster size")
    parser.add_argument("--info", action="store_true", help="Show information about parameters without running the simulation")
    return parser.parse_args()

def run_iterative(comm, save, tol):
    p_old, pop = run_sim(comm, height=1, length=10,pres=8,T=.8,num_steps=1000, save=1, tol=.05, 
                         mesh_created=False, meshed=None, new_membrane=True)
    for i in range(1,20):
        if np.max(p_old)+i >80:
            break
        p_old, pop = run_sim(comm, height=1, length=10,pres=8,T=.8,num_steps=1000, save=1, tol=.05, 
             mesh_created=False, meshed=None, new_membrane=False, p_old=p_old, pg=np.max(p_old)+i)
    """
    Run simulations for a range of radii and pressures.

    This function iterates over the given ranges of radii and pressures,
    running a simulation for each combination.

    :param comm: MPI communicator object
    :type comm: mpi4py.MPI.Intracomm
    :param radius_range: Range of radii to simulate
    :type radius_range: numpy.ndarray
    :param pressure_range: Range of pressures to simulate
    :type pressure_range: numpy.ndarray
    """

def show_parameter_info():
    print("## Parameter Information")
    print("This script accepts the following parameters for single run:")
    print("  --radius: The radius for the simulation (float)")
    print("  --length: The length for the simulation (float)")
    print("  --pressure: The pressure for the simulation (float)")
    print("\nExample usage:")
    print("  python your_script.py --radius 0.5 --length 10 --pressure 100")
    print("This script accepts the following parameters for loop run:")
    print("  --length: The length for the simulation (float)")
    print("  --radius_range: The radius for the simulation (start,end,step)")
    print("  --pressure_range: The pressure for the simulation (start,end,step)")


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
    print("save",save)
    print("tol",tol)

    run_iterative(comm, save, tol)


