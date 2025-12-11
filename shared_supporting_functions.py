#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec



def create_TA_Blue_White_Red_colormap(min_max):
    #white_fractional_location is a number between 0 and 1
    TA_colormap = np.ones([100,4])
    white_fractional_location = (0.0 - min_max[0])/ (min_max[1] - min_max[0])
    index_white = int(np.round(white_fractional_location*100))

    # setup the blue portion of the colormap
    for i in range(0, index_white):
        TA_colormap[i,0] = i/index_white
        TA_colormap[i,1] = i/index_white

    # setup the red portion of the colormap
    for i in range(index_white, 100):
        TA_colormap[i,1] = (100-i)/(100-index_white)
        TA_colormap[i,2] = (100-i)/(100-index_white)

    colorlist = TA_colormap.tolist()
    cmap_name = "TA_blue_white_red_colormap"
    TA_colormap = LinearSegmentedColormap.from_list(cmap_name, colorlist, N=120)
    return TA_colormap




def create_TA_Blue_White_Red_Black_colormap(min_max):
    #white_fractional_location is a number between 0 and 1
    TA_colormap = np.ones([100,4])
    white_fractional_location = (0.0 - min_max[0])/ (min_max[1] - min_max[0])
    index_white = int(np.round(white_fractional_location*100))

    # setup the blue portion of the colormap
    for i in range(0, index_white):
        TA_colormap[i,0] = i/index_white
        TA_colormap[i,1] = i/index_white

    # setup the red portion of the colormap
    full_red_index = int((100 - index_white)*2/4 + index_white)
    for i in range(index_white, full_red_index):
        TA_colormap[i,1] = (full_red_index-i)/(full_red_index-index_white)
        TA_colormap[i,2] = (full_red_index-i)/(full_red_index-index_white)
    for i in range(full_red_index, 100):
        TA_colormap[i,1] = 0.0
        TA_colormap[i,2] = 0.0

    # setup the red-to-black portion of the colormap
    for i in range(full_red_index, 100):
        TA_colormap[i,0] = (100-i)*3/4 / (100-full_red_index) + 0.25

    colorlist = TA_colormap.tolist()
    cmap_name = "TA_blue_white_red_colormap"
    TA_colormap = LinearSegmentedColormap.from_list(cmap_name, colorlist, N=120)
    return TA_colormap




def get_TA_matrix(TA_matrix):
    ## Handle the input if it's a raw matrix or if it's a filename that needs to be passed to a function to load a raw matrix
    if  np.array([TA_matrix[-5:] == '.hdf5' , TA_matrix[-5:] == '.HDF5',  TA_matrix[-3:] == '.h5',  TA_matrix[-3:] == '.H5' ]).any() :
        TA_matrix = load_hdf5_data(TA_matrix, 'Average')
    elif TA_matrix[-4:] == '.csv':
        TA_matrix = np.loadtxt(TA_matrix, delimiter=',',  ndmin=2)
        
    return TA_matrix







def get_TA_probe_counts(TA_matrix):
    ## Handle the input if it's a raw matrix or if it's a filename that needs to be passed to a function to load a raw matrix
    if  np.array([TA_matrix[-5:] == '.hdf5' , TA_matrix[-5:] == '.HDF5',  TA_matrix[-3:] == '.h5',  TA_matrix[-3:] == '.H5' ]).any() :
        TA_matrix = load_hdf5_data(TA_matrix, 'Spectra/Sweep_0_Probe_Spectrum')
    elif TA_matrix.find('.hdf5') != -1:
        TA_matrix = load_hdf5_data( TA_matrix[0:TA_matrix.find('.hdf5') + 5], 'Spectra/Sweep_0_Probe_Spectrum')
    elif TA_matrix.find('.HDF5') != -1:
        TA_matrix = load_hdf5_data( TA_matrix[0:TA_matrix.find('.HDF5') + 5], 'Spectra/Sweep_0_Probe_Spectrum')
    elif TA_matrix.find('.h5') != -1:
        TA_matrix = load_hdf5_data( TA_matrix[0:TA_matrix.find('.h5') + 3], 'Spectra/Sweep_0_Probe_Spectrum')
    elif TA_matrix.find('.H5') != -1:
        TA_matrix = load_hdf5_data( TA_matrix[0:TA_matrix.find('.H5') + 3], 'Spectra/Sweep_0_Probe_Spectrum')
    else:
        print('Didnt recognize file format.')    
    
    return TA_matrix






def load_hdf5_data(filename,dataset_path_string):
    
    # the dataset_path_string uses / to delimit the groups and subgroups and datasets, e.g. 'experiment_1/readings/voltage' 
    with h5py.File(filename, 'r') as f:
        # 1. Access the dataset object (this is a pointer to the file, not the data yet)
        dset = f[dataset_path_string][:]
    
    # 2. Convert to NumPy array
    # The [:] slice syntax tells h5py to read the whole dataset into memory
    return dset





def list_hdf5_contents(HDF5_filename):
    """
    Opens an HDF5 file and prints its internal structure,
    including groups, datasets, dimensions, and data types.
    """
    
    # Check if file exists in current directory
    if not os.path.exists(HDF5_filename):
        print(f"Error: The file '{HDF5_filename}' was not found in the current directory.")
        return

    try:
        with h5py.File(HDF5_filename, 'r') as f:
            print(f"\nStructure of file: {HDF5_filename}")
            print("=" * 60)

            def print_attrs(name, obj):
                """
                Callback function for visititems.
                name: The full path name of the object (e.g., 'group/subgroup/dataset')
                obj: The actual HDF5 object (Group or Dataset)
                """
                # Calculate indentation level based on how deep the object is
                # 'group/dataset' has 1 slash, so it's level 1, etc.
                level = name.count('/')
                indent = '    ' * level
                
                # Get just the name of the current item (remove parent path)
                item_name = name.split('/')[-1]

                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 Group: {item_name}")
                
                elif isinstance(obj, h5py.Dataset):
                    # For datasets, we print shape (dims) and data type
                    print(f"{indent}📄 Dataset: {item_name}")
                    print(f"{indent}    => Shape: {obj.shape}")
                    print(f"{indent}    => Type:  {obj.dtype}")

            # visititems iterates over every object in the file recursively
            f.visititems(print_attrs)
            print("=" * 60)

    except Exception as e:
        print(f"Could not read file: {e}")

