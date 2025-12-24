# pyTAgui
A python, matplotlib, and QT based GUI for transient absorption processing.

# Installation
1: Install python preferably via anaconda (e.g. miniconda) 

2: Create of activate a conda environment that has installed: matplotlib, h5py, PyQt5, numpy, scipy, and any other necessary modules.

3: Clone or fork or download the python files to a directory where you keep your software

4: On Mac or Linux: add the folder to your $PYTHONPATH via .bashrc (or similar). On Windows: add the folder via System Properties -> Advanced -> Environment Variables -> PYTHONPATH

    e.g.
    export PYTHONPATH="/path/to/your/custom/modules:$PYTHONPATH"

3:  Change the working directory of your python terminal to the folder that has your TA data files.
    import os
    os.chdir('path/to/data/files')

4:  You now have two options for executing the code:
    4a. from a linux/macosx terminal, e.g., 
        python -m TA_plot_matrix 'HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5'
        python -m TA_t0_correction_and_background_removal 'HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5'
    4b. open an instance of python and import, e.g.,
        import os
        import sys
        import shared_functions_classes as TA_sh                
        import TA_plot_matrix as TA_plt                         
        import TA_merge_matrices as TA_mrg                      
        import TA_t0_correction_and_background_removal as TA_t0    
        TA_plot_matrix.TA_plot_matrix_app(my_filename)

5: If you want to make any changes or fixes to the Python files, you will need to refresh the Python module cache, by e.g.,
    import importlib
    importlib.reload(TA_plt)



# List of main functions
1: shared_supporting_functions              usually abbreviated as TA_sh

2: TA_plot_matrix                           usually abbreviated as TA_plt

3: TA_merge_matrices                        usually abbreviated as TA_mrg

4: TA_t0_correction_and_background_removal  usually abbreviated as TA_t0



# List of Shared Supporting Functions
1: TA_sh.TA_matrix_window_average(TA_matrix,window_size)

2: TA_sh.create_TA_Blue_White_Red_colormap(min_max)

3: TA_sh.create_TA_Blue_White_Red_Black_colormap(min_max)

4: TA_sh.list_hdf5_contents(HDF5_filename)

5: TA_sh.load_hdf5_data(filename,dataset_path_string)



# Example Work Flow
TA_blue_spectrum_hdf5_filename = 'HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5'

TA_red_spectrum_hdf5_filename  = 'HHHF_Zn_heme_ZnCl_p425nm_red_300uW.h5'

TA_t0.t0_correction_and_background_removal(TA_blue_spectrum_hdf5_filename)

TA_t0.t0_correction_and_background_removal(TA_red_spectrum_hdf5_filename)

TA_mrg.merge_TA_matrices(TA_blue_spectrum_hdf5_filename+'.t0_corr.csv.merged.csv',TA_red_spectrum_hdf5_filename+'.t0_corr.csv.merged.csv')




# Next Code To Develop
o Develop a SVD amd Global Analysis module 


# File Locations
/...BACKED_UP/Software/...Transient_Absorption_Processing/python_qt_TA_data_processing_GUI

https://github.com/damonturney/pyTAgui


