from mpi4py import MPI

# dolfinx and meshing
import numpy as np

from dx_sim import run_sim

import argparse


def parse_range(arg):
    try:
        start, end, step = map(float, arg.split(','))
        return start, end, int(step)
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be start,end,step")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters or in a loop.")
    parser.add_argument("--mode", choices=["single", "loop"], required=True, help="Run mode: single run or loop")
    parser.add_argument("--radius", type=float, help="Radius for single run")
    parser.add_argument("--length", type=float, help="Length for single run")
    parser.add_argument("--pressure", type=float, help="Pressure for single run")
    parser.add_argument("--radius_range", type=parse_range, help="Radius range for loop (start,end,steps)")
    parser.add_argument("--pressure_range", type=parse_range, help="Pressure range for loop (start,end,steps)")
    parser.add_argument("--info", action="store_true", help="Show information about parameters without running the simulation")
    return parser.parse_args()

def run_loop_simulation(comm, radius_range, pressure_range):
    for r in radius_range:
        for p in pressure_range:
            run_sim(comm, height=1,length=10,pres=p,T=.8,num_steps=800,r=r)
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

    if args.mode == "single":
        if args.radius is None or args.length is None or args.pressure is None:
            raise ValueError("For single mode, radius, length, and pressure must be specified.")
        u_, p_, V, mesh = run_sim(comm, height=1, length=args.length, pres=args.pressure, T=.8, num_steps=800, 
                                  r=args.radius, file=False, run=2, tol=.05)
    elif args.mode == "loop":
        if any(arg is None for arg in [args.radius_start, args.radius_end, args.radius_steps, 
                                       args.pressure_start, args.pressure_end, args.pressure_steps]):
            raise ValueError("For loop mode, all radius and pressure range parameters must be specified.")
        radius_range = np.linspace(args.radius_start, args.radius_end, args.radius_steps)
        pressure_range = np.linspace(args.pressure_start, args.pressure_end, args.pressure_steps)
        run_loop_simulation(comm, radius_range, pressure_range)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=231,T=.8,num_steps=500,r=.75,file=False,run=2, tol=0.05)
    #for r in np.linspace(0.58,0.75,3):
    #    for p in np.linspace(2,230, 15):
    #        u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=p,T=.8,num_steps=800,r=r,file=False,run=2, tol=0.05)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=213.71,T=.8,num_steps=800,r=.01,file=False,run=2, tol=0.05)
    #u_, p_, V, mesh = run_sim(comm, height=1,length=10,pres=230,T=.8,num_steps=800,r=.01,file=False,run=2, tol=0.05)

