from mpi4py import MPI

# dolfinx and meshing
import numpy as np
import matplotlib.pyplot as plt
from dx_sim import run_sim
from dx_utils import create_obst

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters or in a loop.")
    parser.add_argument("--mode", choices=["single", "loop"], required=True, help="Run mode: single run or loop")
    parser.add_argument("--radius", type=float, help="Radius for single run")
    parser.add_argument("--length", type=float, help="Length for single run")
    parser.add_argument("--save", type=int, help="Save data to dtool db")
    parser.add_argument("--tol", type=float, help="Mesh raster size")
    parser.add_argument("--pressure", type=float, help="Pressure for single run")
    parser.add_argument("--radius_range", nargs='+',type=float, help="Radius range for loop (start,end,steps)")
    parser.add_argument("--pressure_range", nargs='+', type=int, help="Pressure range for loop (start,end,steps)")
    parser.add_argument("--info", action="store_true", help="Show information about parameters without running the simulation")
    return parser.parse_args()

def run_loop_simulation(comm, radius_range, pressure_range, save, tol):
    res = np.empty((len(radius_range), len(pressure_range) ))
    for i, r in enumerate(radius_range):
        mesh = list(create_obst(comm,1, 10, r, 5, tol))
        for j, p in enumerate(pressure_range):
            # parameters: (comm, height=1, length=3,pres=8,T=.5,num_steps=500,r=0, save=False, tol=.05):
            res[i,j] = run_sim(comm, height=1,length=10,pres=p,T=.8,num_steps=1600,r=r,save=save,tol=tol, meshed=mesh)
    np.savetxt("mfl_arr.txt",res, fmt='%.10e')
    plt.imshow(res)
    plt.savefig("mfl_imshow.pdf")
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
    if args.mode == "single":
        if args.radius is None or args.length is None or args.pressure is None:
            raise ValueError("For single mode, radius, length, and pressure must be specified.")
        u_, p_, V, mesh = run_sim(comm, height=1, length=args.length, pres=args.pressure, T=.8, num_steps=1600, 
                                  r=args.radius, save=save, tol=tol)

    elif args.mode == "loop":
        if any(arg is None for arg in [args.radius_range, args.pressure_range,args.length]):
            raise ValueError("For loop mode, length and all radius and pressure range parameters must be specified.")
        if len(args.radius_range) == 3:
            start, end, step = args.radius_range
            radius_range = np.linspace(start, end, int(step))
        if len(args.pressure_range) == 3:
            start, end, step = args.pressure_range
            pressure_range = np.linspace(start, end, step)
        else:
            print("Provide start stop end values for pressure and radius!")
        run_loop_simulation(comm, radius_range, pressure_range, save, tol)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=231,T=.8,num_steps=500,r=.75,file=False,run=2, tol=0.05)
    #for r in np.linspace(0.58,0.75,3):
    #    for p in np.linspace(2,230, 15):
    #        u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=p,T=.8,num_steps=800,r=r,file=False,run=2, tol=0.05)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=213.71,T=.8,num_steps=800,r=.01,file=False,run=2, tol=0.05)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=230,T=.8,num_steps=800,r=.01,file=False,run=2, tol=0.05)

