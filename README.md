# pyTAgui
/Users/damon/Desktop/BACKED_UP/Software/DamonWrittenSoftware/Transient_Absorption_Processing/python_qt_TA_data_processing_GUI

# Installation
1: Install python preferably via anaconda (e.g. miniconda) 

2: Install matplotlib, h5py, numpy, scipy, and any other necessary modules (easiest with conda or mamba install)

3: Clone or fork or download the python files to a directory where you usually keep your software

4: On Mac or Linux machines add the software folder to your PYTHONPATH via .bashrc (or similar); on windows machines use System Properties -> Advanced -> Environment Variables -> PYTHONPATH

    add the following to .bashrc (or similar) on mac/linux, or add the path manually on windows machines
    export PYTHONPATH="/path/to/your/custom/modules:$PYTHONPATH"

3:  Change the working directory of your python terminal to the folder that has your TA data files.
    import os
    os.chdir('path/to/data/files')

4: After doing step 3 you need to import the TA processing modules into your python instance.
    import os
    import sys
    import shared_supporting_functions as TA_sh             # abbreviation of TA_shared
    import plot_TA_matrix as TA_plt                         # abbreviation of TA_plot
    import merge_TA_matrices as TA_mrg                      # abbreviation of TA_merge
    import t0_correction_and_background_removal as TA_t0    # abbreviation of TA_remove_t0_and_background

5: If you want to make any changes or fixes to the Python files, you will need to refresh the Python module cache, by e.g.,
    import importlib
    importlib.reload(TA_plt)
        or
    quitting and restarting python



# List of main functions
1: shared_supporting_functions             usually abbreviated as TA_sh

2: plot_TA_matrix                          usually abbreviated as TA_plt

3: merge_TA_matrices                       usually abbreviated as TA_mrg

4: t0_correction_and_background_removal    usually abbreviated as TA_t0


List of Shared Supporting Functions, with modules named as above.

1: TA_sh.create_TA_Blue_White_Red_colormap(min_max)

2: TA_sh.create_TA_Blue_White_Red_Black_colormap(min_max)

3: TA_sh.list_hdf5_contents(HDF5_filename)

4: TA_sh.load_hdf5_data(filename,dataset_path_string)




# Example Work Flow
TA_blue_spectrum_hdf5_filename = 'HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5'
TA_red_spectrum_hdf5_filename  = 'HHHF_Zn_heme_ZnCl_p425nm_red_300uW.h5'

TA_t0.t0_correction_and_background_removal(TA_blue_spectrum_hdf5_filename)
TA_t0.t0_correction_and_background_removal(TA_red_spectrum_hdf5_filename)
TA_mrg.merge_TA_matrices(TA_blue_spectrum_hdf5_filename+'.t0_corr.csv.merged.csv',TA_red_spectrum_hdf5_filename+'.t0_corr.csv.merged.csv')




# Next Code To Develop
o Develop a SVD amd Global Analysis module 


