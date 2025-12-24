#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline

# PyQt5 Imports (The GUI Library)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QLabel, QPushButton, QStatusBar, QGroupBox, QComboBox)
from PyQt5.QtCore import Qt

# Matplotlib Imports (The Plotting Library)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT 
from matplotlib.figure import Figure

# Import the shared functions and classes
import shared_functions_classes as TA_sh


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column


class TA_merge_matrix_GUI():   
    def __init__(self, TA_matrix_input_filenames):
        self.app = QApplication(sys.argv)                                           # Create the QApplication instance           
        self.TA_matrix_input_filenames = TA_matrix_input_filenames
        self.main_pyqt5_window = TA_merge_matrix_GUI_main_Window(self.TA_matrix_input_filenames)   # Create the main pyqt5 window
        self.main_pyqt5_window.show()                                               # Display the window
        sys.exit(self.app.exec())                                                   # Start the event loop        



class TA_merge_matrix_GUI_main_Window(QMainWindow):    #This nomenclature causes this class to inherit all the methods and properties of the QMainWindow class from PyQT5.
    def __init__(self, TA_matrix_input_filenames):
        print('NOTE: This function always merges TA data into the format of the first TA_matrix in the input list.')

        super().__init__() #This executes a lot of code that builds the PyQT5 window using the QMainWindow class.
        
        self.TA_matrix_input_filenames = TA_matrix_input_filenames

        # pyqt5 window setup: needs the QMainWindow object to be inherited into this class. 
        self.setWindowTitle("TA matrix merger")
        self.dpi = QApplication.primaryScreen().logicalDotsPerInch() #Get screen DPI
        window_taller_factor = 1.2
        self.window_width_inches = 15
        self.window_height_inches = 10.8 * window_taller_factor
        self.resize(int(self.window_width_inches * self.dpi), int(self.window_height_inches * self.dpi))  # Width, Height in pixels (converted from inches)

        # Create the main pyqt5 object (Qwidget) and then create the geometric layout that will hold a LHS box (for the matplotlib stuff) and RHS box (for the pyqt5 Controls).
        self.main_window_widget = QWidget()
        self.setCentralWidget(self.main_window_widget)
        self.horiz_pyqt5_layout = QHBoxLayout(self.main_window_widget) # Horizontal: Plot on Left, Controls on Right

        # Add the vertical pyqt5 layout to the horizontal (i.e. the main) layout. Create the geometric layout of the LHS box of the QHBoxLayout window: to hold the matplotlib toolbar on top and the matplotlib figure below. This ensures the toolbar sits right on top of the plot
        self.vertical_layout_widget = QWidget()
        self.vert_pyqt5_layout = QVBoxLayout(self.vertical_layout_widget)   # Vertical pyqt5 layout: toolbar on top, canvas below
        self.vert_pyqt5_layout.setContentsMargins(0, 0, 0, 0)               # Remove extra spacing
        self.horiz_pyqt5_layout.addWidget(self.vertical_layout_widget, stretch=4)

        # Create the Control Panel (aka the pyqt5 layout for inserting buttons and textboxes) 
        self.control_panel = QWidget()
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        self.control_panel_layout.setAlignment(Qt.AlignTop)
        self.horiz_pyqt5_layout.addWidget(self.control_panel, stretch=1) # Add the control panel to main horizontal layout (since it's added to the main horizontal layout 1st, it's located on the right side).

        # Create matplotlib Figure and PyQT5 Canvas (actually, it's a pyqt5 imitation of a matplotlib toolbar). 
        self.fig_han = Figure(dpi=self.dpi)                 # We don't need to specify figsize here since the canvas will auto-scale to fill the pyqt5 layout space.
        self.canvas = FigureCanvasQTAgg(self.fig_han)
        self.toolbar = Custom_QT5_Toolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QLabel { font-family: 'Courier New', Consolas, monospace; font-size: 13pt; white-space: pre; }") # We have to use specific fonts to get the toolbar message to not wiggle with different number font widths.
        self.vert_pyqt5_layout.addWidget(self.toolbar)
        self.vert_pyqt5_layout.addWidget(self.canvas, stretch=4) # Add Canvas to the main horizontal layout (Takes up most of the space. Since it's added to the main horizontal laout 2nd, it's on the left side).

        # Load the TA matrix data and TA non-pumped probe counts 
        self.TA_matrix = TA_sh.get_TA_matrix(TA_matrix_input_filenames[0])    # Loads the TA matrix data from file
        self.presently_selected_idx = 0                                                         # Initialize Data Variables
        self.TA_matrix_wavelengths = self.TA_matrix[0,1:]                                       # Extract the wavelengths and time delays
        self.TA_matrix_delay_times = self.TA_matrix[1:,0]
        self.TA_data = self.TA_matrix[1:,1:]                                                    # Crop the TA image down to remove the wavelengths and delay times
        self.TA_data_rows, self.TA_data_cols = self.TA_data.shape
        self.TA_matrix_interpolator = RegularGridInterpolator((self.TA_matrix_delay_times, np.flip(self.TA_matrix_wavelengths)), np.fliplr(self.TA_data), bounds_error=False, fill_value=None) # Create interpolator for the status bar readout

        # Create the Matplotlib Figure and Canvas
        self.pcolormesh_transects = TA_sh.embedded_matplotlib_pcolormesh_transects(parent_pyqt5_QMainWindow=self, frac_horiz=1.0, frac_vert=1/window_taller_factor)

        # Add simple instructions
        self.fig_han.text(0.79, 0.068, 'Instructions:')
        self.fig_han.text(0.79, 0.03, 'Double mouse click (left button)\nto move crosshairs.')

        # Group Box to Choose the TA_matrix for merge inspection
        choose_TA_matrix_grp = QGroupBox("Merge & Probe Controls")
        choose_TA_matrix_grp.setStyleSheet('QLabel { font-size: 10pt; } QLineEdit { font-size: 10pt; } QPushButton { font-size: 10pt; } QComboBox { font-size: 10pt; }')
        choose_TA_matrix = QGridLayout()
        choose_TA_matrix.setVerticalSpacing(4)
        choose_TA_matrix.setContentsMargins(2, 5, 2, 2)
        choose_TA_matrix.addWidget(QLabel("Select TA dataset:"), 0, 0)          # TA Dataset Dropdown list
        self.combo_datasets = QComboBox()
        self.combo_datasets.addItems(self.TA_matrix_input_filenames)
        self.combo_datasets.currentIndexChanged.connect(self.update_TA_matrix_plot_via_dropdown)
        self.combo_datasets.setFixedWidth(150)
        choose_TA_matrix.addWidget(self.combo_datasets, 1, 0)
        choose_TA_matrix_grp.setLayout(choose_TA_matrix)
        self.control_panel_layout.addWidget(choose_TA_matrix_grp)

         # Group Box of Buttons/Textboxes for the merge
        merge_grp = QGroupBox("Merge & Probe Controls")
        merge_grp.setStyleSheet('QLabel { font-size: 10pt; } QLineEdit { font-size: 10pt; } QPushButton { font-size: 10pt; } QComboBox { font-size: 10pt; }')
        merge_layout = QGridLayout()
        merge_layout.setVerticalSpacing(4)
        merge_layout.setContentsMargins(2, 5, 2, 2)       
        merge_layout.addWidget(QLabel("Ignore range min (nm):"), 2, 0)              # Zero Out Probe Counts textbox
        self.txt_zero_range_min = QLineEdit("0.0")
        merge_layout.addWidget(self.txt_zero_range_min, 2, 1)
        merge_layout.addWidget(QLabel("Ignore range max (nm):"), 3, 0)              # Zero Out Probe Counts textbox
        self.txt_zero_range_max = QLineEdit("0.0")
        merge_layout.addWidget(self.txt_zero_range_max, 3, 1)
        self.btn_zero_counts = QPushButton("Zero the Ignore Range")               # Zero Out Counts Button
        self.btn_zero_counts.clicked.connect(self.zero_probe_counts)
        merge_layout.addWidget(self.btn_zero_counts, 4, 0, 1, 2)            # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        self.btn_smooth = QPushButton("Smooth Merge Fraction")              # Smooth Merge Fraction Button
        self.btn_smooth.clicked.connect(self.smooth_the_merge)              # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        merge_layout.addWidget(self.btn_smooth, 5, 0, 1, 2)                 # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        self.btn_merge = QPushButton("Merge TA Datasets")                   # Merge Button
        self.btn_merge.clicked.connect(self.merge_TA_data)
        merge_layout.addWidget(self.btn_merge, 6, 0, 1, 2)                  # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        merge_grp.setLayout(merge_layout)
        self.control_panel_layout.addWidget(merge_grp)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Load probe counts for each experiment. Calculate probe fractions for merging
        self.all_probe_counts = np.zeros((len(self.TA_matrix_input_filenames), len(self.TA_matrix_wavelengths)))  # Initialize storage
        print('WARNING: The minimum probe count will be subtracted from each probe spectrum to clear background noise.')
        for i, filename in enumerate(self.TA_matrix_input_filenames):
            raw_wavelength = np.flip(TA_sh.get_TA_matrix(filename)[0,1:])                                           # Load raw data
            raw_probe_counts = np.flip(TA_sh.get_TA_probe_counts(filename))
            raw_probe_count_interpolator = CubicSpline(raw_wavelength, raw_probe_counts, extrapolate=True)          # Interpolate to match the master wavelength axis
            self.all_probe_counts[i,:] = raw_probe_count_interpolator(self.TA_matrix_wavelengths)
            self.all_probe_counts[i,:] = self.all_probe_counts[i,:] - np.min(self.all_probe_counts[i,:])            # Subtract minimum to clear background noise
        self.sqrt_all_probe_counts = np.sqrt(self.all_probe_counts)
        self.fraction_in_merge_each_probe_spectrum = self.sqrt_all_probe_counts.copy()
        sum_sqrt = np.sum(self.sqrt_all_probe_counts, axis=0)                                                       # Calculate sum of square roots across all datasets
        sum_sqrt[sum_sqrt == 0] = 1e-9                                                                              # Prevent divide by zero
        for i in range(len(self.TA_matrix_input_filenames)):
            self.fraction_in_merge_each_probe_spectrum[i,:] = self.sqrt_all_probe_counts[i,:] / sum_sqrt
        
        # Initial plot of the probe counts
        self.probe_counts = self.all_probe_counts[self.presently_selected_idx, :]
        self.ax_probe_counts = self.fig_han.add_axes([0.06, 0.01, 0.66, 0.161], sharex=self.pcolormesh_transects.TA_image_axis_han) # Create axes below the main image. [left bottom width height]
        self.ax_probe_counts.axes.yaxis.set_label_text('probe counts', color='b')
        self.ax_probe_counts.tick_params(axis='y', colors='blue')
        self.line_probe_counts, = self.ax_probe_counts.plot(self.TA_matrix_wavelengths, self.probe_counts, color='b')
        self.ax_probe_counts.set_ylim(0, np.max(self.probe_counts) * 1.1)
        idx = self.pcolormesh_transects.crosshair_v_idx                                                             # Red dot for crosshair tracking.  Note: We access crosshair indices from the shared pcolormesh object
        self.red_dot_probe_counts = self.ax_probe_counts.scatter(self.TA_matrix_wavelengths[idx], self.probe_counts[idx], color='red', s=40, zorder=2)
        self.ax_fraction_in_merge = self.ax_probe_counts.twinx()                                                     # Create a second y-axis that shares the same x-axis
        self.line_fraction_in_merge, = self.ax_fraction_in_merge.plot(self.TA_matrix_wavelengths, self.fraction_in_merge_each_probe_spectrum[self.presently_selected_idx,:], color='k', linestyle=':')
        self.ax_fraction_in_merge.set_ylabel('fraction in merge')
        self.ax_fraction_in_merge.set_ylim(0, 1)
        self.ax_probe_counts.tick_params(labelbottom=False)
        self.ax_probe_counts.grid(True, linestyle=':')
        self.status_bar.showMessage(f"Loaded: {TA_matrix_input_filenames[0]}")

        # Set Focus
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # Initial Draw
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect Matplotlib Events
        self.canvas.mpl_connect('button_press_event', self.on_TA_matrix_mouse_click)
        self.canvas.mpl_connect('key_press_event', self.on_TA_matrix_up_down_left_right)


    def update_probe_count_plot(self):                                                                                  # Updates the secondary plots for probe counts and merge fractions.
            self.probe_counts = self.all_probe_counts[self.presently_selected_idx, :]
            self.line_probe_counts.set_ydata(self.probe_counts)
            self.red_dot_probe_counts.set_offsets([self.TA_matrix_wavelengths[self.pcolormesh_transects.crosshair_v_idx], self.probe_counts[self.pcolormesh_transects.crosshair_v_idx]])
            self.line_fraction_in_merge.set_ydata(self.fraction_in_merge_each_probe_spectrum[self.presently_selected_idx,:])
            self.canvas.draw()


    ############################################
    ####### Callbacks for Interactivity with the PyQt5 Control Buttons and Textboxes


    def on_TA_matrix_up_down_left_right(self, event):
        ### Handles arrow keys.
        # Check boundaries
        if event.key == 'up' and self.pcolormesh_transects.crosshair_h_idx < self.TA_data_rows - 1:
            self.pcolormesh_transects.crosshair_h_idx += 1
        elif event.key == 'down' and self.pcolormesh_transects.crosshair_h_idx > 0:
            self.pcolormesh_transects.crosshair_h_idx -= 1
        elif event.key == 'left' and self.pcolormesh_transects.crosshair_v_idx < self.TA_data_cols - 1:
            self.pcolormesh_transects.crosshair_v_idx += 1
        elif event.key == 'right' and self.pcolormesh_transects.crosshair_v_idx > 0:
            self.pcolormesh_transects.crosshair_v_idx -= 1
        else:
            return # Ignore other keys
        self.pcolormesh_transects.update_crosshair_lines()
        self.pcolormesh_transects.update_transects()
        self.red_dot_probe_counts.set_offsets([self.TA_matrix_wavelengths[self.pcolormesh_transects.crosshair_v_idx], self.probe_counts[self.pcolormesh_transects.crosshair_v_idx]])


    def on_TA_matrix_mouse_click(self, event):
        ### Handles mouse clicks to move crosshairs.
        if event.inaxes != self.pcolormesh_transects.TA_image_axis_han: return
        # We only care about Double Clicks (per your original script)
        if event.dblclick:
            # Update indices
            self.pcolormesh_transects.crosshair_v_idx = int(np.abs(self.TA_matrix_wavelengths - event.xdata).argmin())
            self.pcolormesh_transects.crosshair_h_idx = int(np.abs(self.TA_matrix_delay_times - event.ydata).argmin())
            self.pcolormesh_transects.update_crosshair_lines()
            self.pcolormesh_transects.update_transects()
            self.red_dot_probe_counts.set_offsets([self.TA_matrix_wavelengths[self.pcolormesh_transects.crosshair_v_idx], self.probe_counts[self.pcolormesh_transects.crosshair_v_idx]])
            self.canvas.draw()



    def update_TA_matrix_plot_via_dropdown(self):
        self.presently_selected_idx = self.combo_datasets.currentIndex()                                       # Update the presently selected index
        filename = self.TA_matrix_input_filenames[self.presently_selected_idx]
        self.TA_matrix = TA_sh.get_TA_matrix(filename)                                                          # Reload the TA matrix data from file
        self.TA_data = self.TA_matrix[1:,1:]
        self.TA_matrix_wavelengths = self.TA_matrix[0,1:]                                                       # Extract the wavelengths and time delays
        self.TA_matrix_delay_times = self.TA_matrix[1:,0]
        self.pcolormesh_transects.pclrmsh.set_array(self.TA_data.ravel())                                         # Update the Pcolormesh Data
        self.TA_matrix_interpolator = RegularGridInterpolator((self.TA_matrix_delay_times, np.flip(self.TA_matrix_wavelengths)), np.fliplr(self.TA_data), bounds_error=False, fill_value=None) # Re-create Interpolator
        self.update_probe_count_plot()
        self.pcolormesh_transects.update_transects()
        self.canvas.draw()

        
    def zero_probe_counts(self):                                                                                        #Callback for Zero Out Counts Button. Zeroes out probe counts in the range specified by the QLineEdit.
        try:
            w_range = [float(self.txt_zero_range_min.text()) , float(self.txt_zero_range_max.text())]
            if len(w_range) != 2 or w_range[1] <= w_range[0]:
                self.status_bar.showMessage("Error: Range must be 'min,max' with max > min")
                return
            w_idxs = [np.abs(self.TA_matrix_wavelengths - w_range[0]).argmin() , np.abs(self.TA_matrix_wavelengths - w_range[1]).argmin()]    # Find indices
            self.all_probe_counts[self.presently_selected_idx, np.min(w_idxs):np.max(w_idxs)] = 1e-10                   # Zero out the probe counts in the specified range
            self.sqrt_all_probe_counts[:] = np.sqrt(self.all_probe_counts)                                              # Recalculate square roots
            sum_sqrt = np.sum(self.sqrt_all_probe_counts, axis=0)
            sum_sqrt[sum_sqrt == 0] = 1e-9
            for i in range(len(self.TA_matrix_input_filenames)):
                self.fraction_in_merge_each_probe_spectrum[i,:] = self.sqrt_all_probe_counts[i,:] / sum_sqrt
            self.update_TA_matrix_plot_via_dropdown()                                        # Update the plots
            self.update_probe_count_plot()
            self.status_bar.showMessage(f"Zeroed probe counts between {w_range[0]} and {w_range[1]} nm")
        except ValueError:
            self.status_bar.showMessage("Error: Invalid input. Use format '450,500'")



    def smooth_the_merge(self):                                                                                             #Callback for Smooth the Merge Fraction Button. Smooths the merge fraction for the presently selected dataset.
        try:
            w_range = [float(self.txt_zero_range_min.text()) , float(self.txt_zero_range_max.text())]
            w_idxs = [np.abs(self.TA_matrix_wavelengths - w_range[0]).argmin(), np.abs(self.TA_matrix_wavelengths - w_range[1]).argmin()]
            nm_per_pixel = (np.max(self.TA_matrix_wavelengths) - np.min(self.TA_matrix_wavelengths)) / len(self.TA_matrix_wavelengths) # Determine smoothing window (approx 2nm)
            smoothing_window_size = int(2 / nm_per_pixel)
            if smoothing_window_size < 1: smoothing_window_size = 1   
            for w_idx in w_idxs:                                                                                        # Perform Smoothing
                if (w_idx > 2 * smoothing_window_size) and (w_idx < len(self.TA_matrix_wavelengths) - 2 * smoothing_window_size):   # Avoid edges
                    for _ in range(4):                                                                                  # Repeat smoothing 4 times
                        for ww_idx in range(w_idx - smoothing_window_size, w_idx + smoothing_window_size):
                            start = ww_idx - smoothing_window_size
                            end = ww_idx + smoothing_window_size
                            self.all_probe_counts[self.presently_selected_idx, ww_idx] = np.sum(self.all_probe_counts[self.presently_selected_idx, start:end]) / (2 * smoothing_window_size)
            self.sqrt_all_probe_counts[:] = np.sqrt(self.all_probe_counts)                                              # Recalculate square roots
            sum_sqrt = np.sum(self.sqrt_all_probe_counts, axis=0)
            sum_sqrt[sum_sqrt == 0] = 1e-9
            for i in range(len(self.TA_matrix_input_filenames)):
                self.fraction_in_merge_each_probe_spectrum[i,:] = self.sqrt_all_probe_counts[i,:] / sum_sqrt
            self.update_TA_matrix_plot_via_dropdown()                                                  # Update the plots
            self.status_bar.showMessage("Smoothing applied.")
        except ValueError:
            self.status_bar.showMessage("Error: Invalid input for range.")




    def merge_TA_data(self):                                                                #Merges all datasets based on the calculated fractions and saves to disk.
            self.status_bar.showMessage("Merging data...")
            QApplication.processEvents()
            first_TA_matrix = TA_sh.get_TA_matrix(self.TA_matrix_input_filenames[0])
            merged_TA_data = first_TA_matrix.copy()
            merged_TA_data[1:,1:] = first_TA_matrix[1:,1:] * self.fraction_in_merge_each_probe_spectrum[0,:]    # Initialize with first dataset weighted
            
            final_merged_wavelengths = first_TA_matrix[0,1:]
            final_merged_delay_times = first_TA_matrix[1:,0]
            for i in range(1, len(self.TA_matrix_input_filenames)):                       # Loop through rest
                next_filename = self.TA_matrix_input_filenames[i]
                next_TA_matrix = TA_sh.get_TA_matrix(next_filename)
                next_w = next_TA_matrix[0,1:]
                next_t = next_TA_matrix[1:,0]
                needs_interp = (np.abs(next_w - final_merged_wavelengths) > 0.01).any() or (np.abs(next_t - final_merged_delay_times) > 0.1).any()  # Check if interpolation is needed
                if needs_interp:
                    print(f'Interpolating {next_filename}...')
                    interp_matrix = self.Interpolate_TA_matrix(next_filename, final_merged_wavelengths, final_merged_delay_times)
                    merged_TA_data[1:,1:] += interp_matrix[1:,1:] * self.fraction_in_merge_each_probe_spectrum[i,:]
                else:
                    print(f'Merging {next_filename} (No interpolation needed)...')
                    merged_TA_data[1:,1:] += next_TA_matrix[1:,1:] * self.fraction_in_merge_each_probe_spectrum[i,:]
            try:
                output_filename = list_of_TA_matrix_filenames[0]+'.merged.csv'
                np.savetxt(output_filename, merged_TA_data, delimiter=',')
                self.status_bar.showMessage(f"Merge Complete! Saved to {output_filename}")
                print(f'File saved: {output_filename}')
            except Exception as e:
                self.status_bar.showMessage(f"Save Failed: {e}")


    def Interpolate_TA_matrix(self, TA_matrix_input, new_ws, new_ts):
        TA_matrix = TA_sh.get_TA_matrix(TA_matrix_input)
        old_ws = TA_matrix[0,1:]
        old_ts = TA_matrix[1:,0]
        print('Starting an Interpolation. This will take a minute or two. Be patient.')
        TA_matrix_interpolator = RegularGridInterpolator((old_ts, old_ws), TA_matrix[1:,1:], bounds_error=False, fill_value=None)  # Create the Interpolator: bounds_error=False allows extrapolation or handling points outside the grid without crashing fill_value=None tells it to extrapolate values outside the bounds
        TA_matrix_interpolated = np.zeros([len(new_ts)+1, len(new_ws)+1])
        grid_t, grid_w = np.meshgrid(new_ts, new_ws, indexing='ij') # Create Grids
        TA_matrix_interpolated[0,1:]  = new_ws
        TA_matrix_interpolated[1:,0]  = new_ts
        TA_matrix_interpolated[1:,1:] = TA_matrix_interpolator((grid_t, grid_w ))
        print("Interpolation successful.")
        return TA_matrix_interpolated




class Custom_QT5_Toolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent_pyqt5_QMainWindow, coordinates=True):
        super().__init__(canvas, parent_pyqt5_QMainWindow, coordinates)
        self.parent_pyqt5_QMainWindow = parent_pyqt5_QMainWindow
    def home(self, *args, **kwargs):
        # Call the Original Home Behavior. This performs the actual view reset in Matplotlib
        super().home(*args, **kwargs)
        # Now I can force my own view
        self.parent_pyqt5_QMainWindow.pcolormesh_transects.TA_image_axis_han.set_xlim(self.parent_pyqt5_QMainWindow.pcolormesh_transects.home_xlim)
        self.parent_pyqt5_QMainWindow.pcolormesh_transects.TA_image_axis_han.set_ylim(self.parent_pyqt5_QMainWindow.pcolormesh_transects.home_ylim)
        self.parent_pyqt5_QMainWindow.canvas.draw()




# For Matt Sfeirs TA beamline the hdf5 files hold the probe counts in 'Spectra/Sweep_0_Probe_Spectrum'
# For the Astrella TA system the probe counts are located in 
def create_merge_ratio_between_2_TA_probe_Matrices(TA_probe1_counts_filename, TA_probe2_counts_filename, probe1_blackout, probe2_blackout):
    # Obtain the probe counts for the probe1 
    blue_probe_counts = np.flip( TA_sh.load_hdf5_data(Sfeir_TA_blue_spectrum_hdf5_filename,'Spectra/Sweep_0_Probe_Spectrum') )                     #Obtain the blue probe spectrum 
    blue_probe_wavelengths = np.flip( TA_sh.load_hdf5_data(Sfeir_TA_blue_spectrum_hdf5_filename,'Average')[0,1:] )
    spline_interpolator = CubicSpline(blue_probe_wavelengths, blue_probe_counts, extrapolate = True)
    blue_probe_counts_interpolated = spline_interpolator(final_interpolated_wavelengths)

    # Obtain the probe counts for the red spectrum
    red_probe_counts = np.flip( TA_sh.load_hdf5_data(Sfeir_TA_red_spectrum_hdf5_filename,'Spectra/Sweep_0_Probe_Spectrum') )                     #Obtain the red probe spectrum
    red_probe_wavelengths = np.flip( TA_sh.load_hdf5_data(Sfeir_TA_red_spectrum_hdf5_filename,'Average')[0,1:] )
    spline_interpolator = CubicSpline(red_probe_wavelengths, red_probe_counts, extrapolate = True)
    red_probe_counts_interpolated = spline_interpolator(final_interpolated_wavelengths)

    # Merge the -DT/T data from the blue and red spectra 
    merge_percentage_red = np.sqrt(red_probe_counts_interpolated) / ( np.sqrt(red_probe_counts_interpolated) + np.sqrt(blue_probe_counts_interpolated) ) 
    merge_percentage_red[final_interpolated_wavelengths > 700] = 1       #This is done to avoid including the enormous error in the blue spectrum above 700 nm






# Define a hook to ensure exceptions are caught by the debugger/terminal
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception", file=sys.stderr)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception


if __name__ == "__main__":       # This line is asking: "Is this file the one that started the process?"  We can have only one QApplication instance per process, so we need to ensure this is the main file.
    # Check if a filename was provided in the command line arguments
    if len(sys.argv) > 1:
        TA_matrix_input_filenames = sys.argv[1]
    else:
        # Fallback default if no file is provided
        TA_matrix_input_filenames = ['HHHF_Zn_heme_ZnCl_p425nm_blue_300uW.h5.t0_corr.csv','HHHF_Zn_heme_ZnCl_p425nm_red_300uW.h5.t0_corr.csv']  # default input for testing:
    # Start the application
    window = TA_merge_matrix_GUI(TA_matrix_input_filenames) #The QApplication instance must be created before this.  The init method recognizes QApplication already exists and uses it.
    
