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
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_press(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if "iterative_canal_" not in i and "_100.0" not in i:
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
        if "iterative_canal2_" not in i and "_100.0" not in i:
            continue

        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        #print(u)
        u = np.where((u >= -100) & (u <= 100), u, 0)     
        r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]

        ax.plot(u, label=rf"$P_G={extracted}$")
        arr[r_index, p_index] = np.mean(u)
    print(radii, press)
    print(arr.shape)
    #neg = ax.plot(arr)
    #ax.contour(arr, colors='red')
    ax.set_xticklabels(np.linspace(0,2,7).round(2) )
    #ax.set_yticklabels(radii)
    #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    ax.legend()
    ax.set_ylabel(r'Pressure on membrane', fontsize=14)
    ax.set_xlabel(r'L (Membrane length)', fontsize=14)
    return arr
    fig.savefig("press_dist.pdf", format='pdf', dpi=300, bbox_inches='tight')


arr = plot_press(pat, pat2, "/p_courve_0.32/p_courve.txt", "Pressure Distribution", yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'

def plot_press(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if "iterative_canal_" not in i and "_100.0" not in i:
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
        if "iterative_canal2_" not in i and "_100.0" not in i:
            continue

        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        #print(u)
        u = np.where((u >= -100) & (u <= 100), u, 0)     
        r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]
        print(u.shape)
        ax.plot(u[:,0], label=rf"$P_G={extracted}$")
        arr[r_index, p_index] = np.mean(u)
    #print(radii, press)
    print(arr.shape)
    #neg = ax.plot(arr)
    #ax.contour(arr, colors='red')
    #ax.set_xticklabels(np.linspace(0,2,7).round(2) )
    #ax.set_yticklabels(radii)
    #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    ax.legend()
    ax.set_ylabel(r'Velocity', fontsize=14)
    ax.set_xlabel(r'y (H/L)', fontsize=14)
    #return arr
    fig.savefig("velocity_pofiles.pdf", format='pdf', dpi=300, bbox_inches='tight')


arr = plot_press(pat, pat2, "/y_at_0_0.32/y_at_0.txt", "Pressure Distribution", yl='P delta', legend=True)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'
import time

def plot_press1(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if not re.search(r'80\.0\/data$', i):
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
        if not re.search(r'80\.0\/data$', i):
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        time.sleep(.01)
        #u = np.where((u >= -100) & (u <= 100), u, 0)     
        #r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]
        #print(u.shape)
        print(float(extracted)+.80, u[0])
        ax.scatter(float(extracted)+.80, u[1], label=rf"$P_G={extracted}$")
        #ax.plot(u)
        #arr[r_index, p_index] = u[0] # np.mean(u)
    
    #print(radii, press)
    print(arr.shape)
    #neg = ax.imshow(arr)
    #ax.contour(arr, colors='red')
    #ax.set_xticklabels(np.linspace(0,2,7).round(2) )
    #ax.set_yticklabels(radii)
    #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    ax.legend()
    ax.set_ylabel(r'massflowrate', fontsize=18)
    ax.set_xlabel(r'$P_G$', fontsize=18)
    #return arr
    fig.savefig("mass_flow.pdf", format='pdf', dpi=300, bbox_inches='tight')


arr2 = plot_press1(pat, pat2, "/massflowrate_0.32/massflowrate.txt", "Massflowrate", yl='P delta', legend=True)

# %% jupyter={"outputs_hidden": true}
print(arr)

# %%
pat = r'\d+\.\d+(?=_)'
pat2 = r'\d+\.\d+(?=/)'
import time

def plot_press2(pt, pt2, file, title, yl='Velocity', legend=True, filename="output.pdf"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    radii_list, press_list = [], []
    
    # First pass: collect unique radii and pressure values
    for i in datadir_list:
        if not re.search(r'100\.0\/data$', i):
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
        if not re.search(r'100\.0\/data$', i):
            continue
        match, match2 = re.search(pt, i), re.search(pt2, i)
        print("match", i) if match is None else None
        print("match2", i) if match2 is None else None      
        extracted, extracted2 = float(match.group()), float(match2.group())
        u = np.loadtxt(i+file)
        time.sleep(.01)
        #u = np.where((u >= -100) & (u <= 100), u, 0)     
        #r_index, p_index = np.where(radii == extracted)[0][0], np.where(press == extracted2)[0][0]
        #print(u.shape)
        ax.plot(u[:,0], label=rf"$P_G={extracted}$")
        #arr[r_index, p_index] = u[0] # np.mean(u)
    
    #print(radii, press)
    print(arr.shape)
    #neg = ax.imshow(arr)
    #ax.contour(arr, colors='red')
    #ax.set_xticklabels(np.linspace(0,2,7).round(2) )
    #ax.set_yticklabels(radii)
    #fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
    #if legend:
    ax.legend()
    ax.set_ylabel(r'massflowrate', fontsize=14)
    ax.set_xlabel(r'L (membrane length)', fontsize=14)
    return arr
    #fig.savefig("press_dist.pdf", format='pdf', dpi=300, bbox_inches='tight')


arr2 = plot_press2(pat, pat2, "/y_courve_0.32/y_courve.txt", "Massflowrate", yl='P delta', legend=True)

# %%
