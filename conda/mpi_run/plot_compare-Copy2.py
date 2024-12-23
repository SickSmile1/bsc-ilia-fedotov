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

# %%
rootdir = os.getcwd()

# pat = r'canal_\d+\.\d+_\d+'
pat = r'\d+\.\d+'
path = rootdir+"/dtool_db/iterative_runs/"
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
def plot_res(pt, file, title, yl='Massflowrate in kg/s', fkt=np.mean, legend=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    for i in datadir_list:
        if "_3" in i:
            match = re.search(pt, i)
            extracted = match.group()
            if cu ==0:
                mfl = np.loadtxt(i+file).reshape(1, -1)
                cu += 1
                ax.scatter(float(extracted), fkt(mfl[0,:]))
            else:
                mfl = np.append(mfl, np.loadtxt(i+file).reshape(1, -1), axis=0)
                cu+=1
                ax.scatter(float(extracted),fkt(mfl[cu-1,:]))
    if legend:
        ax.legend()
    ax.set_xlabel(r'radius in r/H', fontsize=14)
    ax.set_ylabel(yl, fontsize=14)


# %%
plot_res(pat, "/massflowrate.txt", "Mean massflowrate vs radius")

# %%
plot_res(pat, "/massflowrate.txt", "L2-Norm massflowrate vs radius", fkt=np.linalg.norm)


# %%
def ret_mid_element(x):
    return x[int(x[0].size/2)]

#plot_res(pat, "/massflowrate.txt", "massflowrate at obstacle vs radius", fkt=ret_mid_element)


# %%
plot_res(pat, "/pressure_avg.txt", "Mean pressure cut vs radius", yl="Pressure in P, p/H")

# %%
plot_res(pat, "/pressure_avg.txt", "L2-Norm pressure cut vs radius", fkt=np.linalg.norm)


# %%
def plot_amp(pt, file, title, yl='Massflowrate in kg/s', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    for i in datadir_list:
        if "3.txt" in i:
            match = re.search(pt, i)
            extracted = match.group()
            if cu ==0:
                mfl = np.loadtxt(i+file)
                pr_delta = np.linspace(80,0,mfl.size)
                pr = np.loadtxt(i+"/pressure_avg.txt")
                cu += 1
                ax.plot(pr, mfl, label=f'r= {extracted:s}')
            elif cu in s:
                mfl = np.loadtxt(i+file)
                pr_delta = np.linspace(80,0,mfl.size)
                pr = np.loadtxt(i+"/pressure_avg.txt")
                cu+=1
                ax.plot(pr, mfl, label=f'r= {extracted:s}')
            cu+=1
        if legend:
            ax.legend()
        ax.set_xlabel(r'pressure in P, p/H', fontsize=14)
        ax.set_ylabel(yl, fontsize=14)

plot_amp(pat,"/massflowrate.txt","Massflowrate over presure drop")


# %%
def plot_amp(pt, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots(2)
    ax[0].set_title(title)
    cu = 0
    for i in datadir_list:
        match = re.search(pt, i)
        extracted = match.group()
        if cu ==0:
            u = np.loadtxt(i+file)
            pr = np.loadtxt(i+"/y_at_.5.txt")
            u2 = np.loadtxt(i+"/x_at_1.txt")
            pr2 = np.loadtxt(i+"/y_at_1.txt")
            cu += 1
            ax[0].plot(u,pr, label=f'r= {extracted:s}')
            ax[1].plot(u2,pr2, label=f'r= {extracted:s}')
        elif cu in s:
            u = np.loadtxt(i+file)
            pr = np.loadtxt(i+"/y_at_.5.txt")
            u2 = np.loadtxt(i+"/x_at_1.txt")
            pr2 = np.loadtxt(i+"/y_at_1.txt")
            cu+=1
            ax[0].plot(u,pr, label=f'r= {extracted:s}')
            ax[1].plot(u2,pr2, label=f'r= {extracted:s}')
        cu+=1
    if legend:
        ax[0].legend()
        ax[1].legend()
    ax[1].set_xlabel(r'Velocity in u = u/U', fontsize=14)
    ax[1].set_ylabel(r'Canal length l in l = L/h', fontsize=14)

plot_amp(pat,"/x_at_.5.txt","Massflowrate over presure drop")

# %%
radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,])
press = np.linspace(20,2300,15)

pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'


for k in datadir_list:
    print(k)
    break
    match = re.search(pat, k)
    match2 = re.search(pat2, k)
    extracted = match.group()
    extracted2 = match2.group()
    print(extracted, extracted2)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,.75])
    press = np.round(np.linspace(20,2300,15),1)
    arr = np.empty((12, 15))
    for i in datadir_list:
        if "parametric_canal" not in i:
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
        if cu ==0:
            u = np.loadtxt(i+file)
            #print(i+file)
            #print(u)
            cu += 1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]

        else:
            if not os.path.exists(i+file):
                print(i+file)
            u = np.loadtxt(i+file)
            #print(i+file)
            #print(u)
            cu+=1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]

    # neg = ax.contour(arr)
    neg = ax.imshow(arr)
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7, label="Massflowrate")
    #if legend:
    #    ax.legend()
    ax.set_ylabel(r'obstacle r', fontsize=14)
    ax.set_xlabel(r'delta P', fontsize=14)
    ax.set_title("Massflowrates L=10,dp=20-2300")
    return arr

ret = plot_mfl(pat, pat2, "/massflowrate_0.60/massflowrate.txt", "Massflowrate", yl='P delta', legend=True)


# %%
pt = r'\d+\.\d+(?=_)'
pt2 = r'\d+\.\d+(?=/)'
k = "/home/sick/Documents/GIT/bsc-ilia-fedotov/conda/mpi_run/dtool_db1/parametric_canal_0.02_500.0/data/massflowrate_0.60/massflowrate.txt"
i = "/home/sick/Documents/GIT/bsc-ilia-fedotov/conda/mpi_run/dtool_db1/parametric_canal_0.02_500.0/data"
match = re.search(pt, i)
match2 = re.search(pt2, i)
extracted = match.group()
extracted2 = match2.group()
print(extracted, extracted2)
print(os.path.exists(k))
np.loadtxt(k)

# %%
ret.shape

# %%
radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,.75])
press = np.round(np.linspace(20,2300,15),1)
plt.plot(press, ret.T, label=radii)
plt.legend()

# %% jupyter={"source_hidden": true}
radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,.75])
press = np.round(np.linspace(20,2300,15),1)
plt.plot(radii, ret, label=press)
plt.legend()

# %% jupyter={"source_hidden": true}
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,.75])
    press = np.round(np.linspace(20,2300,15),1)
    arr = np.empty((12, 15))
    for i in datadir_list:
        if "parametric_canal" not in i:
            continue
        match = re.search(pt, i)
        match2 = re.search(pt2, i)
        extracted = match.group()
        extracted2 = match2.group()
        if cu ==0:
            u = np.loadtxt(i+file+"")
            cu += 1
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

        else:
            u = np.loadtxt(i+file)
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            cu+=1
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

    neg = ax.imshow(arr)
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    #ax.set_xlabel(r'obstacle r', fontsize=14)
    #ax.set_ylabel(r'delta P', fontsize=14)

plot_mfl(pat, pat2, "/y_at_5_0.60/y_at_5.txt", "Velocity", yl='P delta', legend=True)

# %% jupyter={"source_hidden": true}
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.12,.17,.22,.27,.33,.41,.50,.58,.67,.75])
    press = np.round(np.linspace(20,2300,15),1)
    arr = np.empty((12, 15))
    for i in datadir_list:
        #print(i+file)
        match = re.search(pt, i)
        match2 = re.search(pt2, i)
        extracted = match.group()
        extracted2 = match2.group()
        #print(extracted)
        #print(extracted2)
        if cu ==0:
            u = np.loadtxt(i+file+"")
            #print(i)
            #print(u)
            cu += 1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

        else:
            u = np.loadtxt(i+file)
            #print(i)
            #print(u)
            cu+=1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

    neg = ax.imshow(arr)
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    #ax.set_xlabel(r'obstacle r', fontsize=14)
    #ax.set_ylabel(r'delta P', fontsize=14)

plot_mfl(pat, pat2, "/y_at_0_0.60/y_at_0.txt", "Velocity", yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.11,.16,.22,.27,.32,.38,.43,.48,.54,.59,.64,.70,.75])
    press = np.round(np.linspace(50,2250,15),1)
    arr = np.empty((15, 15))
    for i in datadir_list:
        if "parametric2_canal" not in i:
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
        if cu ==0:
            u = np.loadtxt(i+file)
            #print(i+file)
            #print(u)
            cu += 1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]

        else:
            if not os.path.exists(i+file):
                print(i+file)
            u = np.loadtxt(i+file)
            #print(i+file)
            #print(u)
            cu+=1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = u[0]

    # neg = ax.contour(arr)
    neg = ax.imshow(arr)
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7, label="Massflowrate")
    #if legend:
    #    ax.legend()
    ax.set_ylabel(r'obstacle r', fontsize=14)
    ax.set_xlabel(r'delta P', fontsize=14)
    ax.set_title("Massflowrates L=20,dp=50-2250")
    return arr

ret = plot_mfl(pat, pat2, "/massflowrate_0.60/massflowrate.txt", "Massflowrate", yl='P delta', legend=True)


# %%
radii = np.array([.01,.06,.11,.16,.22,.27,.32,.38,.43,.48,.54,.59,.64,.70,.75])
press = np.round(np.linspace(50,2250,15),1)
plt.plot(press, ret.T, label=radii)
plt.legend()

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.11,.16,.22,.27,.32,.38,.43,.48,.54,.59,.64,.70,.75])
    press = np.round(np.linspace(50,2250,15),1)
    arr = np.empty((15, 15))
    for i in datadir_list:
        if "parametric2_canal" not in i:
            continue
        match = re.search(pt, i)
        match2 = re.search(pt2, i)
        extracted = match.group()
        extracted2 = match2.group()
        if cu ==0:
            u = np.loadtxt(i+file+"")
            cu += 1
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

        else:
            u = np.loadtxt(i+file)
            if u[0] > 100 or u[0] < -100:
                u[:] = 0
            cu+=1
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

    neg = ax.imshow(arr)
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    #ax.set_xlabel(r'obstacle r', fontsize=14)
    #ax.set_ylabel(r'delta P', fontsize=14)

plot_mfl(pat, pat2, "/y_at_5_0.60/y_at_5.txt", "Velocity", yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cu = 0
    radii = np.array([.01,.06,.11,.16,.22,.27,.32,.38,.43,.48,.54,.59,.64,.70,.75])
    press = np.round(np.linspace(50,2250,15),1)
    arr = np.empty((15, 15))
    for i in datadir_list:
        #print(i+file)
        if "parametric2_canal" not in i:
            continue
        match = re.search(pt, i)
        match2 = re.search(pt2, i)
        extracted = match.group()
        extracted2 = match2.group()
        #print(extracted)
        #print(extracted2)
        if cu ==0:
            u = np.loadtxt(i+file+"")
            #print(i)
            #print(u)
            cu += 1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

        else:
            u = np.loadtxt(i+file)
            #print(i)
            #print(u)
            cu+=1
            #print(np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0])
            arr[np.where(radii==float(extracted))[0], np.where(press==float(extracted2))[0]] = np.max(u)

    neg = ax.imshow(arr)
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    #ax.set_xlabel(r'obstacle r', fontsize=14)
    #ax.set_ylabel(r'delta P', fontsize=14)

plot_mfl(pat, pat2, "/y_at_0_0.60/y_at_0.txt", "Velocity", yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'
plt.rcParams.update({'font.size': 18})
def plot_mfl(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if "parametric2_canal" not in i:
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    
    # Second pass: fill the arr
    for i in datadir_list:
        if "parametric2_canal" not in i:
            continue

        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        
        u = np.loadtxt(i+file)
        u = np.where((u >= -100) & (u <= 100), u, 0)     
        r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]
       
        arr[r_index, p_index] = u[2]

    neg = ax.imshow(arr)
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press/10).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    ax.set_ylabel(r'obstacle r', fontsize=18)
    ax.set_xlabel(r'delta P', fontsize=18)
    fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    return arr, radii, press

arr, radii, press = plot_mfl(pat, pat2, "/flux_trapz_0.10/flux_trapz.txt", "Massflowrate", yl='P delta', legend=True,filename="flux_1.pdf")



# %%
arr1, radii1, press1=plot_mfl(pat, pat2, "/flux_trapz_0.60/flux_trapz.txt", "Massflowrate", yl='P delta', legend=True, filename="flux_6.pdf")

# %%
res, radii, press=plot_mfl(pat, pat2, "/massflowrate_0.60/massflowrate.txt", "Massflowrate", yl='P delta', legend=True, filename="mfl_6.pdf")

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'
plt.rcParams.update({'font.size': 14})
def plot_press(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if "parametric2_canal" not in i:
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    
    # Second pass: fill the arr
    for i in datadir_list:
        if "parametric2_canal" not in i:
            continue

        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        #u = np.where((u >= -100) & (u <= 100), u, 0)     
        r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]
       
        arr[r_index, p_index] = u[5]

    neg = ax.imshow(arr)
    ax.contour(arr, colors='red')
    ax.set_xticks(np.arange(len(press)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels((press/10).astype(int))
    ax.set_yticklabels(radii)
    fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    ax.set_ylabel(r'obstacle r', fontsize=14)
    ax.set_xlabel(r'delta P', fontsize=14)
    fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')

plot_press(pat, pat2, "/p_at_0_0.10/p_at_0.txt", "Pressure", yl='P delta', legend=True,filename="pressure_1.pdf")


# %%
_= plot_press(pat, pat2, "/p_at_5_0.10/p_at_5.txt", "Pressure", yl='P delta', legend=True,filename="pressure_6.pdf")


# %%
_=plot_press(pat, pat2, "/p_at_5_0.50/p_at_5.txt", "Pressure", yl='P delta', legend=True,filename="pressure_6.pdf")


# %%
_=plot_press(pat, pat2, "/p_at_1_0.50/p_at_1.txt", "Pressure", yl='P delta', legend=True,filename="pressure_end_6.pdf")


# %%
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(press, arr.T, label=radii)
ax.legend(frameon=False)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(press1, arr1.T, label=radii1)
ax.legend(frameon=False)
plt.show()

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
x = np.linspace(0,2,100)
memb = -zetta(8, 1.9553851318e+03, 8.2760372899e+02, 1.9553851318e+03/4, 2, x)
plt.plot(x,1+memb*.01)
plt.show()


# %%

# %%
def define_membrane(factory, begin, end, l_tenth, l1, lc1):
    x = np.linspace(0,2,100)
    memb = -zetta(8, 1.9553851318e+03, 8.2760372899e+02, 1.9553851318e+03/4, 2, x)
    lines = []
    points = []
    for i in range(1,len(x)-1,len(x)-1):
        points.append(factory.addPoint(x[i-1],memb[i-1],0,lc1/10))
        points.append(factory.addPoint(x[i],memb[i],0, lc1/10))
        lines.append(factory.addLine(points[i-1],points[i]))
    lines.append(factory.addLine(begin,x[0]))
    lines.append(factory.addLine(x[-1],end))
    return lines, points

import gmsh
H=1
L=3
r=.3
Ox=1.5
lc=.03
model_rank = 0
infl, outfl, upper, lower = [],[],[],[]
if True: #comm.rank == model_rank:
    gmsh.initialize()
    gmsh.model.add("canal")
    
    cm = 1 # e-02 # not needed for our sim
    h1 = H * cm
    l1 = L * cm
    r = r * cm
    Lc1 = lc
    l_tenth = L/10 * cm
    
    # We start by defining some points and some lines. To make the code shorter we
    # can redefine a namespace:
    factory = gmsh.model.geo
    model = gmsh.model
    
    factory.addPoint(0, 0, 0, Lc1, 1)
    factory.addPoint(l1, 0, 0, Lc1, 2)
    factory.addPoint(l1, h1 , 0, Lc1, 3)
    factory.addPoint(0, h1, 0, Lc1, 4)
    
    begin = factory.addPoint(Ox-l_tenth, h1, 0, Lc1, 5)
    end = factory.addPoint(Ox+l_tenth, h1, 0, Lc1, 7)
    
    factory.addLine(1, 2, 8)
    factory.addLine(2, 3, 9)
    factory.addLine(3, 7, 10)
    factory.addLine(5, 4, 11)
    factory.addLine(4, 1, 12)
    
    # add obstacle form
    lines, points = define_membrane(factory, begin, end, l_tenth, l1, Lc1)
    # factory.addCircleArc(5, 6, 7, 13)
    
    # Define the inner curve loop (the circle arc)
    factory.addCurveLoop([-13], lines)
    
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
    gmsh.write("mesh.msh")
#infl = comm.bcast(infl, root=0)
#outfl = comm.bcast(outfl, root=0)
#upper = comm.bcast(upper, root=0)
#lower = comm.bcast(lower, root=0)
#gmsh.model = comm.bcast(gmsh.model, root=0)
#mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, comm, model_rank,gdim=2)
#return mesh, ct, ft, infl, outfl, upper, lower

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_press(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if "iterative_canal_" not in i:
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        
        if match and match2:
            radius, pressure = float(match.group()), float(match2.group())
            
            radii_list = radii_list + [radius] if radius not in radii_list else radii_list
            press_list = press_list + [pressure] if pressure not in press_list else press_list


    radii, press = np.sort(np.array(radii_list)), np.sort(np.array(press_list))
    # Create the arr with the correct dimensions
    arr = np.empty((len(radii), len(press)))
    
    # Second pass: fill the arr
    for i in datadir_list:
        if "iterative_canal_" not in i:
            continue

        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        print(u)
        #u = np.where((u >= -100) & (u <= 100), u, 0)     
        r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]

        ax.plot(u)
        #arr[r_index, p_index] = np.mean(u)
    print(radii, press)
    print(arr.shape)
    #neg = ax.plot(arr)
    #ax.contour(arr, colors='red')
    #ax.set_xticklabels((press/10).astype(int))
    #ax.set_yticklabels(radii)
    #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    #    ax.legend()
    ax.set_ylabel(r'pressure on membrane', fontsize=14)
    ax.set_xlabel(r'inlet pressure', fontsize=14)
    #fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')


plot_press(pat, pat2, "/p_courve_0.80/p_courve.txt", "Massflow", yl='P delta', legend=True)

# %%
s = datadir_list[0]
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

match, match2 = re.search(pat, s), re.search(pat2, s)


# %%
match2

# %%
