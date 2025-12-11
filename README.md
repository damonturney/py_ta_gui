# pyTAgui
/Users/damon/Desktop/BACKED_UP/Software/DamonWrittenSoftware/Transient_Absorption_Processing/python_qt_TA_data_processing_v0p5

Installation

1: Save the folder of python files to a directory on your computer at the base of your TA processing work

2: Each time you open a new python instance (aka terminal): add the TA software folder pathname to PYTHONPATH, by opening a python instance and executing the following.

    import sys
    import os
    directory_to_add = 'whichever/path/you/chose'
    directory_to_add = '/Users/damon/Desktop/BACKED_UP/Software/DamonWrittenSoftware/Transient_Absorption_Processing/python_qt_TA_data_processing_v0p5'
    sys.path.append(directory_to_add)

3:  Change the working directory of your python terminal to the folder holding all your TA data files.
    import os
    os.chdir('path/to/data/files')

4: After doing step 3 you need to import the TA processing modules into the python instance.
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









List of main functions with modules named via:  
    import shared_supporting_functions as TA_sh
    import plot_TA_matrix as TA_plt
    import merge_TA_matrices as TA_mrg
    import t0_correction_and_background_removal as TA_t0
 
1: TA_plt.plot_TA_matrix(TA_matrix)

2: TA_mrg.merge_TA_matrices(list_of_TA_matrix_filenames , final_wavelengths, final_delay_times)
    2a: Interpolate_TA_matrix(TA_matrix, new_ws, new_ts)

3: TA_t0.t0_correction_and_background_removal(TA_matrix)

4: TA_fd.fit_TA_decays(TA_matrix).







List of Shared Supporting Functions, with modules named as above.

1: TA_sh.create_TA_Blue_White_Red_colormap(min_max)

2: TA_sh.create_TA_Blue_White_Red_Black_colormap(min_max)

3: TA_sh.list_hdf5_contents(HDF5_filename)

4: TA_sh.load_hdf5_data(filename,dataset_path_string)






Next Things To Do:
o Fix the merge radio button function to work with nature sees of different dimensions
o Develop a SVD module 





Generic Work Flow
Copy the folder "python_qt_TA_data_processing" into the same directory as the raw data files you will process
start python and execute: run python_qt_TA_data_processing_v0p5/load_all_functions.py




Example Work Flow
Sfeir_HHHF_TA_blue_spectrum_IgorProcessed_filename  = 'HHHF_Zn_Heme_ZnCl_e425nm_blue_300uW_zccm.csv'
Sfeir_HHHF_TA_red_spectrum_IgorProcessed_filename   = 'HHHF_Zn_Heme_ZnCl_e425nm_red_300uW_zccm.csv'
Sfeir_HHHF_TA_blue_spectrum_hdf5_filename           = 'HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5'
Sfeir_HHHF_TA_red_spectrum_hdf5_filename            = 'HHHF_Zn_heme_ZnCl_p425nm_red_300uW.h5'
Sfeir_PHOTO_TA_blue_spectrum_IgorProcessed_filename = 'PHOTO_Zn_Heme_ZnCl_e425nm_blue_300uW_zccm.csv'
Sfeir_PHOTO_TA_red_spectrum_IgorProcessed_filename  = 'PHOTO_Zn_Heme_ZnCl_e425nm_red_300uW_zccm.csv'
Sfeir_PHOTO_TA_blue_spectrum_hdf5_filename          = 'PHOTO_Zn_heme_ZnCl_p425nm_blue_300uW.h5'
Sfeir_PHOTO_TA_red_spectrum_hdf5_filename           = 'PHOTO_Zn_heme_ZnCl_p425nm_red_300uW_2.h5'

# Create your final times and wavelengths to interpolate all matrices to
Sfeir_red_TA_image = np.loadtxt('HHHF_Zn_Heme_ZnCl_e425nm_red_300uW_zccm.csv', delimiter=',', ndmin=2)      # I randomly picked this file to be the one holding the final interpolated times and wavelengths
final_interpolated_wavelengths = Sfeir_red_TA_image[1:,0]
final_interpolated_delay_times = Sfeir_red_TA_image[0,1:]

# Perform the interpolations and merges on HHHF data
HHHF_Sfeir_TA_red_blue_merged_processed = merge_Sfeirs_TA_blue_red_spectra(
    Sfeir_HHHF_TA_blue_spectrum_IgorProcessed_filename, 
    Sfeir_HHHF_TA_red_spectrum_IgorProcessed_filename, 
    Sfeir_HHHF_TA_blue_spectrum_hdf5_filename, 
    Sfeir_HHHF_TA_red_spectrum_hdf5_filename,
    final_interpolated_wavelengths,
    final_interpolated_delay_times)

# Perform the interpolations and merges on PHOTO data
PHOTO_Sfeir_TA_red_blue_merged_processed = merge_Sfeirs_TA_blue_red_spectra(
    Sfeir_PHOTO_TA_blue_spectrum_IgorProcessed_filename, 
    Sfeir_PHOTO_TA_red_spectrum_IgorProcessed_filename, 
    Sfeir_PHOTO_TA_blue_spectrum_hdf5_filename, 
    Sfeir_PHOTO_TA_red_spectrum_hdf5_filename,
    final_interpolated_wavelengths,
    final_interpolated_delay_times)

blue_probe_counts = load_hdf5_data(filename,'Spectra/Sweep_0_Probe_Spectrum')                      Obtain the blue probe spectrum 
blue_probe_counts = load_hdf5_data(filename,'Spectra/Sweep_0_Probe_Spectrum')                      Obtain the red probe spectrum
