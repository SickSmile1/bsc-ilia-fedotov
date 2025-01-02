import dtoolcore
import dtoolcore.utils as utils
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import re
import matplotlib as mpl

def ret_fig_ax(rows=1, cols=1):
    fig_width_pt = 448.13095  # Replace with your document's text width
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    fig, ax = plt.subplots(rows,cols, figsize=( fig_width_in, fig_width_in * 0.618), sharey=True)
    return fig, ax

from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

mpl.rcParams['text.usetex'] = False

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
#                      

rootdir = os.getcwd()

# pat = r'canal_\d+\.\d+_\d+'
pat = r'\d+\.\d+'
path = rootdir+"/dtool_db/"
#print(path)
def get_data_directories(root_dir):
    data_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if root.endswith('/data'):
            data_dirs.append(root)
    return data_dirs

datadir_list = get_data_directories(path)

pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'
import time
def plot_pres(pt, pt2, file, title,ax, yl='Velocity', legend=True):
    #fig, ax = plt.subplots()
    #ax.set_title(title)

    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if not re.search(r'.*(parametric2_canal_[^/]+)/data', i):
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list

    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    arr = np.empty((len(radii), len(press)))
    print(radii, press, arr.shape)
    
    for i in datadir_list:
        if not re.search(r'.*(parametric2_canal_[^/]+)/data', i):
            continue
        # print(i+file)
        match = re.search(pt, i)
        match2 = re.search(pt2, i)
        if match == None:
            print("match",i)
        if match2 == None:
            print("match2",i)
        extracted = match.group()
        extracted2 = match2.group()

        u = np.loadtxt(i+file)

        #print(i+file)
        u = u[(u >= 10) & (u <= 2100)]
        #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
        arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)


    # neg = ax.contour(arr)
    neg = ax.imshow(arr)
    #ax.contour(arr, colors='red')

    x_ticks = np.arange(0, len(press), 3) 
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(press[x_ticks].astype(int))
    
    # For y-axis (radii)
    y_ticks = np.arange(0, len(radii), 2) 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(radii[y_ticks])

    if not legend:
        fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7, label="Pressure")
        ax.set_xlabel(r'    ')
    
    if legend:
        fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        ax.set_ylabel(r'Obstacle r')#, fontsize=14)
        ax.set_xlabel(r'$\Delta P$')#, fontsize=14)

    #ax.set_title("Massflowrates L=10,dp=20-2300")
    plt.tight_layout()
    return ax

fig, ax = ret_fig_ax()
ax = plot_pres(pat, pat2, "/p_at_0_0.10/p_at_0.txt", "Pressure",ax, yl='P delta', legend=True)
fig.savefig("pressure_1.pdf", format='pdf', dpi=300, bbox_inches='tight')
fig, ax2 = ret_fig_ax()
ax2 = plot_pres(pat, pat2, "/p_at_0_0.60/p_at_0.txt", "Pressure",ax2, yl='P delta', legend=False)
fig.savefig("pressure_6.pdf", format='pdf', dpi=300, bbox_inches='tight')