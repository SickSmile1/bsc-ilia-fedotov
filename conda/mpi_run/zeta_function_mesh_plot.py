# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: dolfinenv
#     language: python
#     name: dolfinenv
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def zetta(T, p0, pl, pg, L, x):
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
    pd = p0-pl
    return 1/T * (1/2 * (pg - p0) * x**2 + pd/(6*L) * x**3 - 1/6 * (3*pg - 2*p0 - pl)* L * x )


# %%
#zetta(T, p0, pl, pg, L, x):
for i in np.linspace(651,1300,10):
    x = np.linspace(0,2,10)
    memb = zetta(1000, 330, 220, i, 2, x)
    memb += 1
    plt.plot(x,memb,label=f"{i}")
plt.legend(frameon=False)
plt.show()

print(x[0], x[-1])

# %%
"""def zetta(T, p0, pl, pg, L, x):
    pd = p0-pl
    return 1/T * (1/2 * (pg - p0) * x**2 + pd/(6*L) * x**3 - 1/6 * (3*pg - 2*p0 - pl)* L * x )"""

def define_membrane(factory, begin, end, l1, lc1,L):
    x = np.linspace(0,2,100)
    memb = zetta(250, .8553851318e+03, 6.2760372899e+02, 1.0553851318e+03, 2, x)
    memb += 1
    memb = memb.round(3)
    startpoint = L/2-L/10
    lines = []
    points = []
    points.append(begin)
    for i in range(len(x)):
        new_point = factory.addPoint(startpoint + x[i], memb[i], 0, lc1)
        points.append(new_point)
        lines.append(factory.addLine(points[-2], points[-1]))
    lines.append(factory.addLine(points[-1],end))
    return lines, points

import gmsh
H=1
L=10
r=.3
Ox=1.5
lc=.1
model_rank = 0
infl, outfl, upper, lower = [],[],[],[]
gmsh.initialize()
gmsh.model.add("canal")
#gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
#gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

cm = 1 # e-02 # not needed for our sim
h1 = H * cm
l1 = L * cm
r = r * cm
Lc1 = lc

# We start by defining some points and some lines. To make the code shorter we
# can redefine a namespace:
factory = gmsh.model.occ
model = gmsh.model

lowerleft = factory.addPoint(0, 0, 0, Lc1)
lowerright = factory.addPoint(l1, 0, 0, Lc1)
upperright = factory.addPoint(l1, h1 , 0, Lc1)
upperleft = factory.addPoint(0, h1, 0, Lc1)

begin = factory.addPoint(Ox-L/10, h1, 0, Lc1)
end = factory.addPoint(Ox+L/10, h1, 0, Lc1)

inflow_line = factory.addLine(lowerleft, upperleft)
upper_wall_left = factory.addLine(upperleft, begin)
upper_wall_right = factory.addLine(end, upperright)
lower_wall = factory.addLine(upperright, lowerright)
outflow_line = factory.addLine(lowerright, lowerleft)

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

"""
gmsh.model.mesh.setTransfiniteCurve(upper_wall_right, 40)
gmsh.model.mesh.setTransfiniteCurve(upper_wall_left, 40)
gmsh.model.mesh.setTransfiniteCurve(outflow_line, 10)
gmsh.model.mesh.setTransfiniteCurve(inflow_line, 10)
gmsh.model.mesh.setTransfiniteCurve(lower_wall, 100)"""

gmsh.model.addPhysicalGroup(dim=2, tags=[surface], tag=5, name="Domain")
factory.synchronize()
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.RecombineAll", 0)
gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
#infl = comm.bcast(infl, root=0)
#outfl = comm.bcast(outfl, root=0)
#upper = comm.bcast(upper, root=0)
#lower = comm.bcast(lower, root=0)
#gmsh.model = comm.bcast(gmsh.model, root=0)
#mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
#return mesh, ct, ft, infl, outfl, upper, lower

# %%
gmsh.finalize()

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.sans-serif": ["Arial"],
    "font.cursive": ["Arial"],
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 14,
    "svg.fonttype": "none"
})

def ret_fig_ax(rows=1, cols=1):
    fig_width_pt = 448.13095  # Replace with your document's text width
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    fig, ax = plt.subplots(rows,cols, figsize=( fig_width_in*2, fig_width_in )) #, sharey=True)
    #ax.set_aspect('equal')
    return fig, ax


# %%
fig,ax = ret_fig_ax()

x = np.linspace(0,2,11)
memb = zetta(250, .8553851318e+03, 6.2760372899e+02, 1.0553851318e+03, 2, x)
memb += 1

# Get mesh data
nodes = gmsh.model.mesh.getNodes()
elements = gmsh.model.mesh.getElements()

# Extract 2D triangular elements
triangles = elements[2][1].reshape(-1, 3) - 1
x = nodes[1].reshape(-1, 3)[:, 0]
y = nodes[1].reshape(-1, 3)[:, 1]

# show boundaries
bound_arr = np.concatenate((np.linspace(0,4,41), np.linspace(6,10,41) ))
ax.scatter(bound_arr, np.ones(bound_arr.size), color="red") # top wall
ax.scatter(np.linspace(0,10,101), np.zeros_like(( np.linspace(0,10,101) )), color="purple", label="bottom") # bot wall
ax.scatter(np.linspace(4,6,11),memb, color="red", label="top")

sliced = np.linspace(.1,.9,9)
ax.scatter(np.ones(sliced.size)*3.1, sliced, color="green", label="in") # in
ax.scatter(np.ones(sliced.size)*6.9, sliced, color="blue", label="out" ) # in
ax.legend(bbox_to_anchor=(1.0, 0.96), loc='upper left',frameon=False,handlelength=.5)
ax.triplot(x, y, triangles)
#plt.ylim(-.1,1.1)
ax.set_xlim(3,7)
ax.set_xlabel(r'$\bar{x}$ [H]')
ax.set_ylabel(r'$\bar{y}$ [H]')
#plt.axis('equal')
fig.savefig("mesh.pdf", dpi=300)
plt.show()

# %%
print(memb)

# %%
np.linspace(55, 65, 9, endpoint=True)

# %%
p_arr = np.linspace(12,200,15)

def calc_press(parr, L, x):
    return parr*(L-x)

print(calc_press(p_arr, 10, 5))

# %%
p_arr = np.linspace(12,200,15)
height = np.linspace(0,1,100)
def calc_velo(delta_p,y):
    delta_p = np.array(delta_p).reshape(-1, 1)
    return (1/2*delta_p*y )*(1-y)
res = calc_velo(p_arr,0.5)
res1 = calc_velo(p_arr,height)
np.trapezoid(res1[14], height)

# %%
