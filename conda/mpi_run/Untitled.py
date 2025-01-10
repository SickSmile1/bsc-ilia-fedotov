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
import dtoolcore
import dtoolcore.utils as utils
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import re
from dx_utils import zetta
import itertools
from itertools import permutations
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

# %%
datadir_list

# %%
from dtoolcore import DataSet
import json

def get_height_from_dtool(dataset_uri):
    ## Dataset-level metadata
    dataset = DataSet.from_uri(dataset_uri)
    
    # Load the dataset
    dataset = DataSet.from_uri(dataset_uri)
    
    # Get the 'metadata' annotation
    metadata = dataset.get_annotation("metadata")
    
    # Print the metadata
    res = metadata[0]
    if res["p0"] ==0:
        return 1, np.ones(500)
    y = zetta(res["p0"], res["pl"], res["pg"],2,1000, num=500)
    
    return np.min(y), y

def find_matching_files(directory, pattern):
    regex = re.compile(pattern)
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.search(os.path.join(root, file)):
                matching_files.append(os.path.join(root, file))
    return matching_files

def detect_segments(data, threshold):
    differences = np.abs(np.diff(data))
    discontinuities = np.where(differences > threshold)[0]
    segments = np.split(data, discontinuities + 1)
    return segments

def find_best_order(segments):
    n = len(segments)
    best_order = list(range(n))
    best_score = float('inf')
    
    for perm in itertools.permutations(range(n)):
        reordered = np.concatenate([segments[i] for i in perm])
        score = np.sum(np.abs(np.diff(reordered)))
        if score < best_score:
            best_score = score
            best_order = perm
    
    return best_order

def correct_profile(data, threshold):
    segments = detect_segments(data, threshold)
    best_order = find_best_order(segments)
    corrected = np.concatenate([segments[i] for i in best_order])
    return corrected


#print(get_height_from_dtool("/home/sick/Documents/GIT/bsc-ilia-fedotov/conda/mpi_run/dtool_db/iterative_canal_540.0_560.0"))

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

from label_lines import label_lines
#label_lines(plt.gca().get_lines(),xvals=(0.9,0.9,0.9),rotations=(0,0,0))

def plot_y(pt, pt2, file, title, ax,yl='Velocity', legend=True, filename="output.pdf"):
    #fig, ax = ret_fig_ax()
    #ax.set_title(title)
    pg = "550.0"    

    for i in datadir_list:
        if not re.search(r'' + re.escape(pg) + r'\/data$', i):
            continue

        directory = os.path.dirname(i)
        file_pattern = os.path.basename(file)
        matching_files = find_matching_files(directory, file_pattern)
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        h_max, _ = get_height_from_dtool(re.sub(r'\/data$', r'', i))
        file_path = matching_files[0]
        u = np.loadtxt(file_path)
        #rolled_back = correct_profile(u[:,0], .4)
        ax.plot(np.linspace(0,h_max,100),u[:,0], label=rf"${extracted}$")

    handles, labels = ax.get_legend_handles_labels()
    extracted_values = [float(re.search(r'\s*(\d+(?:\.\d+)?)', label).group(1)) for label in labels]

    # Create sorted pairs of (handle, label) based on extracted values
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: float(re.search(r'\s*(\d+(?:\.\d+)?)', x[1]).group(1)))
    
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    lines = ax.get_lines()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lines)))
    for handle, color in zip(sorted_handles, colors):
        handle.set_color(color)
    
    ax.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.0, 0.96), loc='upper left',frameon=False,handlelength=.5)
    ax.text(1.03, 1., r'$P_G$ [$\frac{\mu U}{H}$]', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    #plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    ax.set_ylabel(r'Velocity $\bar{U}$ [U]', fontsize=14)
    ax.set_xlabel(r'Height $\bar{y}$ [H]', fontsize=14)

    fig.savefig("fixed_velocity_pofiles.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    return arr, ax

fig, ax = ret_fig_ax()
arr, ax = plot_y(pat, pat2, "/v_at_5_.*_0.16/y_at_5_.*.txt", "Pressure Distribution",ax, yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

from label_lines import label_lines
#label_lines(plt.gca().get_lines(),xvals=(0.9,0.9,0.9),rotations=(0,0,0))

def plot_press(pt, pt2, file, title,legend=True, filename="output.pdf"):
    fig, ax = ret_fig_ax()
    #ax.set_title(title)
    pg = "740.0"    

    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    print(radii, press, arr.shape)

    
    for i in datadir_list:
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]

    #handles, labels = plt.gca().get_legend_handles_labels()
    #extracted_values = [float(re.search(r'\s*(\d+(?:\.\d+)?)', label).group(1)) for label in labels]

    # Create sorted pairs of (handle, label) based on extracted values
    #sorted_pairs = sorted(zip(handles, labels), key=lambda x: float(re.search(r'\s*(\d+(?:\.\d+)?)', x[1]).group(1)))
    
    #sorted_handles, sorted_labels = zip(*sorted_pairs)
    #plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.0, 1), loc='upper left',frameon=False)
    
    neg = ax.imshow(arr, cmap="plasma")
    ax.contour(arr, colors='red')
    #ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7, label="Massflowrate")
    ax.set_ylabel(r'$P_g$', fontsize=14)
    ax.set_xlabel(r'$\Delta P_{DS}$', fontsize=14)

    plt.show()
    fig.savefig("massflow_membrane.pdf", format='pdf', dpi=300, bbox_inches='tight')
    return radii, press, arr


radii, press, arr = plot_press(pat, pat2, "/flux_trapz_0.32/flux_trapz.txt", "Massflowrate", legend=True)

# %%
p_arr = np.linspace(55,65,9)
height = np.linspace(0,1,100)
fig, ax = ret_fig_ax()
def calc_velo(delta_p,y):
    delta_p = np.array(delta_p).reshape(-1, 1)
    return (1/2*delta_p*y )*(1-y)
res = calc_velo(p_arr,0.5)
res1 = calc_velo(p_arr,height)
arr2 = []
for i in res1:
    arr2.append(np.trapezoid(i, height))
ax.plot(arr2, label="analytical")
ax.plot(arr[0],label="simulated")
ax.set_xticks(np.arange(len(p_arr)))
#ax.set_xticklabels((p_arr).astype(int))
ax.set_xticklabels([f'{p*10:.1f}' for p in p_arr])
ax.set_ylabel("Mass flow rate Q [UH]")
ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]')
plt.legend(frameon=False)
fig.savefig("ana_vs_sim_mfl.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("abs mean dist: ",np.mean( np.abs(arr[0] - arr2) ) )
print("linalg norm: ",np.linalg.norm(arr[0] - arr2))

def calculate_percentage_deviation(array1, array2):
    return 100 * (array2 - array1) / ((array2 + array1) / 2)

print(calculate_percentage_deviation(arr[0], arr2))

# %%
c = (25*(0.8/1000))/(7/100)
c1 = (20*(1/2000))/.03

print(c,c1)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_press_2(pt, pt2, file,ax, yl='Velocity', legend=True, filename="output.pdf", cmap='YlGn'):
    #fig, ax = ret_fig_ax()
    #ax.set_title(title)
    pg = "740.0"    

    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    # print(radii, press, arr.shape)

    
    for i in datadir_list:
        directory = os.path.dirname(i)
        file_pattern = os.path.basename(file)
        matching_files = find_matching_files(directory, file_pattern)
             
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        if legend:
            u = np.loadtxt(i+file)
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]
        else:
            file_path = matching_files[0]
            u = np.loadtxt(file_path)
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)            

    if legend:
        #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        ax.set_ylabel(r'$P_G$ [$\frac{\mu U}{H}$]')#, fontsize=14)
        ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]')#, fontsize=14)
    neg = ax.imshow(arr,cmap=cmap, aspect="auto")
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)

    #plt.show()
    #fig.savefig("velocities_membrane.pdf", format='pdf', dpi=300, bbox_inches='tight')
    return neg, radii, press, arr

def plot_y1(pt, pt2, file,ax, yl='Velocity', legend=True, filename="output.pdf", cmap='YlGn'):
    #fig, ax = ret_fig_ax()
    #ax.set_title(title)
    pg_l = ["_0.0","_700.0","_1000.0"]
    psd = ["550.0/","650.0/"]

    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if not (any(re.search(rf"{re.escape(pg_value)}(?=_|\b)", i) for pg_value in pg_l)):
            # and any(re.search(rf"{re.escape(psd_val)}", i) for psd_val in psd)):
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)

        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    # print(radii, press, arr.shape)

    
    for i in datadir_list:
        if not (any(re.search(rf"{re.escape(pg_value)}(?=_|\b)", i) for pg_value in pg_l)):
            # and any(re.search(rf"{re.escape(psd_val)}", i) for psd_val in psd)):
            continue
        print(i)
        directory = os.path.dirname(i)
        file_pattern = os.path.basename(file)
        matching_files = find_matching_files(directory, file_pattern)
        
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        file_path = matching_files[0]
        if legend:
            u = np.loadtxt(i+file)
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u) #u[0]
        else:
            u = np.loadtxt(file_path)
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)            

    if legend:
        #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        ax.set_ylabel(r'$P_G$ [$\frac{\mu U}{H}$]')#, fontsize=14)
        ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]')#, fontsize=14)
    if not legend:
        #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        ax.set_ylabel(r'Velocity $\bar{U}$ [U]')#, fontsize=14)
        ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]')
    neg = ax.plot(arr.T, label=[r' $P_G$='+str(r) for r in radii])
    #ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    #ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    #ax.set_yticklabels(radii)

    #plt.show()
    #fig.savefig("velocities_membrane.pdf", format='pdf', dpi=300, bbox_inches='tight')
    return neg, radii, press, arr

# %%
fig, ax = ret_fig_ax(1,2)
neg, radii, press, arr = plot_press_2(pat, pat2, "/flux_trapz_0.32/flux_trapz.txt",ax[0], yl='P delta', legend=True, cmap="plasma")
neg1, radii1, press1, arr1 = plot_y1(pat, pat2, "/y_at_5_.*_0.32/y_at_5_.*.txt",ax[1], yl='P delta', legend=False, cmap='YlGn')
#plt.tight_layout()
ax[0].text(-0.2, 1.0, '(a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top')
ax[1].text(-0.1, 1.0, '(b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top')
ax[1].legend(frameon=False)
#fig.colorbar(neg1, ax=ax[1], location='right', anchor=(0, 0.5), shrink=0.75, label="Velocity")
fig.colorbar(neg, ax=ax[0], location='right', anchor=(0, 0.5), shrink=0.75, label="Mass flow rate Q [UH]")
fig.savefig("velocities_massflow_membrane.pdf", format='pdf', dpi=300, bbox_inches='tight')

# %%
p_arr = np.linspace(55,65,9)
height = np.linspace(0,1,100)
fig, ax = ret_fig_ax()
def calc_velo(delta_p,y):
    delta_p = np.array(delta_p).reshape(-1, 1)
    return (1/2*delta_p*y )*(1-y)
res = calc_velo(p_arr,0.5)
res1 = calc_velo(p_arr,height)
#arr2 = []
#for i in res1:
#    arr2.append(np.trapezoid(i, height))
ax.plot(res, label="analytical")
ax.plot(arr1[0],label="simulated")
ax.set_xticks(np.arange(len(p_arr)))
#ax.set_xticklabels((p_arr).astype(int))
ax.set_xticklabels([f'{p:.1f}' for p in p_arr])
ax.set_ylabel("Mass flow rate Q [UH]")
ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]')
plt.legend(frameon=False)
fig.savefig("ana_vs_sim_v.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("abs mean dist: ",np.mean( np.abs(arr1[0] - res) ) )
print("linalg norm: ",np.linalg.norm(arr1[0] - res))

def calculate_percentage_deviation(array1, array2):
    return 100 * (array2 - array1) / ((array2 + array1) / 2)

print(calculate_percentage_deviation(arr1[0], res))

# %%
(1/2*55*.5 )*(1-.5)

# %%
#plt.plot(arr1.T)
#print(arr1.T)
arr1

# %%
a,b,c,d = 5.82, 5.98, 6.1,6.23


# %%
max_h = [0.78,0.71,0.66,0.76,0.68,1,0.73,0.86,0.83,0.81]
max_h = np.abs(np.sort(max_h)[::-1]-1)

max_h_formatted = [f'{x:.2f}' for x in max_h]
fig, ax =ret_fig_ax()
lines = ax.plot(press, arr.T, labelsufficient detail to reproduce?=max_h_formatted)
ax.set_ylabel("Mass flow rate Q [UH]")
#plt.gca().set_yticks(np.arange(len(radii)))
#plt.gca().set_yticklabels((radii).astype(int))
ax.set_xlabel(r'$P_{DS}$ [$\frac{\mu U}{H}$]')
colors = plt.cm.rainbow(np.linspace(0, 1, max_h.size))
for line, color in zip(lines, colors):
    line.set_color(color)

ax.text(1.02, 1., 'Membrane min [H]', transform=ax.transAxes, fontsize=14, va='top')
ax.legend(frameon=False,bbox_to_anchor=(1.0, .95), loc='upper left')
fig = plt.gcf()
fig.savefig("mass_flow_lin_55-65.pdf", format='pdf', dpi=300, bbox_inches='tight')

# %%
arr.T[:,0]

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

#label_lines(plt.gca().get_lines(),xvals=(0.9,0.9,0.9),rotations=(0,0,0))

def plot_ymax(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = ret_fig_ax(1,2)
    plt.subplots_adjust(wspace=.4)
    pg = "650.0"
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if not re.search(r'' + re.escape(pg) + r'\/data$', i):
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
        if not re.search(r'' + re.escape(pg) + r'\/data$', i):
            continue

        directory = os.path.dirname(i)
        file_pattern = os.path.basename(file)
        matching_files = find_matching_files(directory, file_pattern)
        
        h_max, y = get_height_from_dtool(re.sub(r'\/data$', r'', i))
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        print(h_max)
        ax[0].plot(np.linspace(0,2,500),y, label=extracted)
    
    handles, labels = ax[0].get_legend_handles_labels()
    extracted_values = [float(re.search(r'\s*(\d+(?:\.\d+)?)', label).group(1)) for label in labels]

    # Create sorted pairs of (handle, label) based on extracted values
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: float(re.search(r'\s*(\d+(?:\.\d+)?)', x[1]).group(1)))
    
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    lines = ax[0].get_lines()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lines)))
    for handle, color in zip(sorted_handles, colors):
        handle.set_color(color)
    ax[0].legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.0, 0.96), loc='upper left',frameon=False,handlelength=.5)

    ax[0].set_ylabel(r'Height $\bar{y}$ $[H]$', fontsize=14)
    ax[0].set_xlabel(r'Lenght $\bar{x}$ $[H]$', fontsize=14)
    ax[0].set_ylim(0.5,1.01)
    ax[0].set_xlim(0,2)

    ax[0].text(1.03, 1., r'$P_G$ [$\frac{\mu U}{H}$]', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top')
    ax[0].text(-0.15, 1., '(a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top')
    ax[1].text(-0.15, 1., '(b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top')
    return radii, press, arr, ax


radii, press, arr, ax = plot_ymax(pat, pat2, "/y_at_5_.*_0.16/y_at_5_.*.txt", r"Velocity at different $P_g$", yl='P delta', legend=True)
arr2, ax[1] = plot_y(pat, pat2, "/v_at_5_.*_0.16/y_at_5_.*.txt", "Pressure Distribution",ax[1], yl='P delta', legend=True)
#ax[0].text(-0.15, 1.1, '(a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top')
#ax[1].text(-0.15, 1.1, '(b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top')

fig = ax[0].get_figure()
fig.savefig("y_velo.pdf", format='pdf', dpi=300, bbox_inches='tight')

# %%
print(arr[2,:])

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

from label_lines import label_lines
#label_lines(plt.gca().get_lines(),xvals=(0.9,0.9,0.9),rotations=(0,0,0))

def plot_prs(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(8,6))
    #ax.set_title(title)
    pg = "550.0"    
    
    for i in datadir_list:
        if not re.search(r'' + re.escape(pg) + r'\/data$', i):
            continue

        #directory = os.path.dirname(i)
        #file_pattern = os.path.basename(file)
        #matching_files = find_matching_files(directory, file_pattern)
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        h_max, _ = get_height_from_dtool(re.sub(r'\/data$', r'', i))
        #file_path = matching_files[0]
        u = np.loadtxt(i+file)
        #rolled_back = correct_profile(u[:,0], .4)
        ax.plot(np.linspace(0,2,100),u, label=rf"${extracted}$")

    handles, labels = plt.gca().get_legend_handles_labels()
    extracted_values = [float(re.search(r'\s*(\d+(?:\.\d+)?)', label).group(1)) for label in labels]

    # Create sorted pairs of (handle, label) based on extracted values
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: float(re.search(r'\s*(\d+(?:\.\d+)?)', x[1]).group(1)))
    
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.0, .96), loc='upper left',frameon=False)
    
    ax.text(1.13, 1., r'$P_G$ [$\frac{\mu U}{H}$]', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    ax.set_ylabel(r'Pressure $\bar{p}$ [$\frac{\mu U}{H}$]', fontsize=14)
    ax.set_xlabel(r'y [H]', fontsize=14)

    plt.show()
    fig.savefig("pressure_profiles.pdf", format='pdf', dpi=300, bbox_inches='tight')


arr = plot_prs(pat, pat2, "/p_at_5_0.16/p_at_5.txt", "Pressure Distribution", yl='P delta', legend=True)


# %%

def plot_press2(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = ret_fig_ax()
    #ax.set_title(title)
    pg = "740.0"    

    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        #if not re.search(r'' + re.escape(pg) + r'\/data$', i):
        #    continue
        
        #print(i)
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    print(radii, press, arr.shape)

    
    for i in datadir_list:
        #if not re.search(r'' + re.escape(pg) + r'\/data$', i):
        #    continue

        directory = os.path.dirname(i)
        file_pattern = os.path.basename(file)
        matching_files = find_matching_files(directory, file_pattern)
        
        h_max = get_height_from_dtool(re.sub(r'\/data$', r'', i))
        
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        file_path = matching_files[0]
        u = np.loadtxt(file_path)
        arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

    #handles, labels = plt.gca().get_legend_handles_labels()
    #extracted_values = [float(re.search(r'\s*(\d+(?:\.\d+)?)', label).group(1)) for label in labels]

    # Create sorted pairs of (handle, label) based on extracted values
    #sorted_pairs = sorted(zip(handles, labels), key=lambda x: float(re.search(r'\s*(\d+(?:\.\d+)?)', x[1]).group(1)))
    
    #sorted_handles, sorted_labels = zip(*sorted_pairs)
    #plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.0, 1), loc='upper left',frameon=False)
    
    neg = ax.imshow(arr)
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7, label=r'Pressure $\bar{p}$ [$\frac{\mu U}{H}$]')
    ax.set_ylabel(r'$P_G$ [$\frac{\mu U}{H}$]', fontsize=14)
    ax.set_xlabel(r'$\Delta P_{SD}$ [$\frac{\mu U}{H}$]', fontsize=14)

    plt.show()
    fig.savefig("press_membrane.pdf", format='pdf', dpi=300, bbox_inches='tight')
    return radii, press, arr

radii, press, arr = plot_press2(pat, pat2, "/p_at_5_0.32/p_at_5.txt", r"Velocity at different $P_g$", yl='P delta', legend=True)

# %%
arr

# %%
