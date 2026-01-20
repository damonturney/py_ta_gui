#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

# PyQt5 Imports (The GUI Library)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QLabel, QPushButton, QStatusBar, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata

# Matplotlib Imports (The Plotting Library)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT 
from matplotlib.figure import Figure

# Import the shared functions and classes
import src.pyTAgui.shared_functions_classes as TA_sh


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column


class TA_t0_correction_and_background_removal():   
    def __init__(self, TA_matrix_input):
        self.app = QApplication(sys.argv)                                           # Create the QApplication instance           
        self.TA_matrix_input = TA_matrix_input
        self.main_pyqt5_window = TA_t0_correction_and_background_removal_main_Window(self.TA_matrix_input)   # Create the main pyqt5 window
        self.main_pyqt5_window.show()                                               # Display the window
        sys.exit(self.app.exec())                                                   # Start the event loop        



class TA_t0_correction_and_background_removal_main_Window(QMainWindow):    #This nomenclature causes this class to inherit all the methods and properties of the QMainWindow class from PyQT5.
    def __init__(self, TA_matrix_input):
        
        super().__init__() #This executes a lot of code that builds the PyQT5 window using the QMainWindow class.
        
        # pyqt5 window setup: needs the QMainWindow object to be inherited into this class. 
        self.setWindowTitle("TA t0 Correction and Background Removal")
        self.dpi = QApplication.primaryScreen().logicalDotsPerInch() #Get screen DPI
        self.window_width_inches = 15
        self.window_height_inches = 10.5
        self.resize(int(self.window_width_inches * self.dpi), int(self.window_height_inches * self.dpi))  # Width, Height in pixels (converted from inches)

        # Create the main pyqt5 object (Qwidget) and then create the geometric layout that will hold a LHS box (for the matplotlib stuff) and RHS box (for the pyqt5 Controls).
        self.main_window_widget = QWidget()
        self.setCentralWidget(self.main_window_widget)
        self.horiz_pyqt5_layout = QHBoxLayout(self.main_window_widget) # Horizontal: Plot on Left, Controls on Right

        # Add the vertical pyqt5 layout to the horizontal (main) layout. Create the geometric layout of the LHS box of the QHBoxLayout window: to hold the matplotlib toolbar on top and the matplotlib figure below. This ensures the toolbar sits right on top of the plot
        self.vertical_layout_widget = QWidget()
        self.vert_pyqt5_layout = QVBoxLayout(self.vertical_layout_widget)   # Vertical pyqt5 layout: toolbar on top, canvas below
        self.vert_pyqt5_layout.setContentsMargins(0, 0, 0, 0) # Remove extra spacing
        self.horiz_pyqt5_layout.addWidget(self.vertical_layout_widget, stretch=4)

        # Create the Control Panel (aka the pyqt5 layout for inserting buttons and textboxes) 
        self.control_panel = QWidget()
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        self.control_panel_layout.setAlignment(Qt.AlignTop)
        self.horiz_pyqt5_layout.addWidget(self.control_panel, stretch=1) # Add the control panel to main horizontal layout (since it's added to the main horizontal layout 1st, it's located on the right side).

        # Create matplotlib Figure and PyQT5 Canvas (actually, it's a pyqt5 imitation of a matplotlib toolbar). 
        self.fig_han = Figure(dpi=self.dpi)   # We don't need to specify figsize here since the canvas will auto-scale to fill the pyqt5 layout space.
        self.canvas = FigureCanvasQTAgg(self.fig_han)
        self.toolbar = Custom_QT5_Toolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QLabel { font-family: 'Courier New', Consolas, monospace; font-size: 13pt; white-space: pre; }") # We have to use specific fonts to get the toolbar message to not wiggle with different number font widths.
        self.vert_pyqt5_layout.addWidget(self.toolbar)
        self.vert_pyqt5_layout.addWidget(self.canvas, stretch=4) # Add Canvas to the main horizontal layout (Takes up most of the space. Since it's added to the main horizontal laout 2nd, it's on the left side).

        # Load the TA matrix data
        self.TA_matrix_input = TA_matrix_input
        self.TA_matrix = TA_sh.get_TA_matrix(self.TA_matrix_input)
        # Extract the wavelengths and time delays
        self.TA_matrix_wavelengths = self.TA_matrix[0,1:]
        self.TA_matrix_delay_times = self.TA_matrix[1:,0]
        # Crop the TA image down to remove the wavelengths and delay times
        self.TA_data = self.TA_matrix[1:,1:]
        self.TA_data_rows, self.TA_data_cols = self.TA_data.shape
        # Create interpolator for the status bar readout
        self.TA_matrix_interpolator = RegularGridInterpolator((self.TA_matrix_delay_times, np.flip(self.TA_matrix_wavelengths)), np.fliplr(self.TA_data), bounds_error=False, fill_value=None)

        # Create the Matplotlib Figure and Canvas
        self.pcolormesh_transects = TA_sh.embedded_matplotlib_pcolormesh_transects(parent_pyqt5_QMainWindow=self, frac_horiz=1.0, frac_vert=1.0)

        # Add simple instructions
        self.fig_han.text(0.76, 0.068, 'Instructions:')
        self.fig_han.text(0.76, 0.03, 'Double mouse click (left button)\nto move crosshairs.')

        # create pyqt5 controls for t0 correction and background removal
        t0_grp_box = QGroupBox("t0 Correction Controls")
        t0_grp_box.setStyleSheet('QLabel { font-size: 10pt; } QLineEdit { font-size: 10pt; } QPushButton { font-size: 10pt; }')
        t0_grid_layout = QGridLayout()
        t0_grid_layout.setVerticalSpacing(2)
        t0_grid_layout.setContentsMargins(2, 5, 2, 2)
        self.btn_add_t0 = QPushButton("Add point for t0 fit")
        self.btn_add_t0.clicked.connect(self.add_t0_point)
        t0_grid_layout.addWidget(self.btn_add_t0, 0, 0)
        # Button: Fit t0 line
        self.btn_fit_t0 = QPushButton("Fit t0 Line")
        self.btn_fit_t0.clicked.connect(self.fit_t0)
        t0_grid_layout.addWidget(self.btn_fit_t0, 1, 0)
        # Button: Correct t0 & Background
        self.btn_correct = QPushButton("Correct t0 & Background")
        self.btn_correct.clicked.connect(self.correct_t0_background)
        t0_grid_layout.addWidget(self.btn_correct, 2, 0)
        # Button: Save Corrected Matrix
        self.btn_save_matrix = QPushButton("Save Corrected Matrix")
        self.btn_save_matrix.clicked.connect(self.save_TA_matrix)
        t0_grid_layout.addWidget(self.btn_save_matrix)
        t0_grp_box.setLayout(t0_grid_layout)
        self.control_panel_layout.addWidget(t0_grp_box)

        # Initialize t0 correction variables
        # Create a list to hold a history of clicked points
        self.click_history = [[-10000,-10000]]    # start with a dummy click to allow yourself to check if values are increasing, then cut away this zero for final fitting
        self.t0_each_wavelength = np.zeros(len(self.TA_matrix_wavelengths))

        # Create plot objects for t0 visualization (Line for fit, Scatter for points)
        # We access the axis handle that resides inside the pcolormesh_transects object
        self.t0_points_scatter = self.pcolormesh_transects.TA_image_axis_han.scatter([], [], color='white', s=30, edgecolor='k', zorder=10)
        self.t0_fit_line, = self.pcolormesh_transects.TA_image_axis_han.plot([], [], linewidth=0.7, alpha=0.7, color='k', zorder=11)
        self.t0_fit_line.set_visible(False)
        self.t0_fit_line.set_linewidth(1.0)        

        # Set focus so keyboard works immediately
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # Initial Draw
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect Matplotlib Events
        self.canvas.mpl_connect('button_press_event', self.on_TA_matrix_mouse_click)
        self.canvas.mpl_connect('key_press_event', self.on_TA_matrix_up_down_left_right)

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
            if (event.button == 3):
                self.add_t0_point()


    def add_t0_point(self):
            #Adds the current crosshair location to the list of points for t0 fitting.
            # Get current indices from the transect object
            w_idx = self.pcolormesh_transects.crosshair_v_idx
            t_idx = self.pcolormesh_transects.crosshair_h_idx
            
            # Get actual values
            w_val = self.TA_matrix_wavelengths[w_idx]
            t_val = self.TA_matrix_delay_times[t_idx]
            
            # Add to history
            self.click_history.append([w_val, t_val])
            
            # Update the visual scatter plot of selected points
            self.t0_fit_line.set_visible(True)
            self.t0_points_scatter.set_offsets(np.array(self.click_history))
            self.canvas.draw()
            
            self.status_bar.showMessage(f"Added t0 point: {w_val:.1f} nm, {t_val:.2f} ps")


    def fit_t0(self):
        #Fits a cubic spline to the selected t0 points.
        click_history = np.array(self.click_history[1:])     # cut away the first click_history data because it was artificial
        if len(click_history) < 2: # Need more than 2 for the fit
            self.status_bar.showMessage("Error: Select at least 2 points to fit t0.") 
            return
        # Sort points by wavelength (required for spline)
        click_history = click_history[click_history[:, 0].argsort()]
        try:
            # Create spline fit
            t0_spline_interpolator = CubicSpline(click_history[:, 0], click_history[:, 1], extrapolate=True)
            self.t0_each_wavelength[:] = t0_spline_interpolator(self.TA_matrix_wavelengths)
            # Update the fit line on the plot
            self.t0_fit_line.set_data(self.TA_matrix_wavelengths, self.t0_each_wavelength)
            self.t0_fit_line.set_visible(True)
            self.canvas.draw()
            self.status_bar.showMessage("t0 fit updated.")
        except Exception as e:
            self.status_bar.showMessage(f"Fit failed: {e}")


    def correct_t0_background(self):
        #Performs the t0 chirp correction, then background removal.
        if np.all(self.t0_each_wavelength == 0):
            self.status_bar.showMessage("Error: Please fit t0 first.")
            return
        self.status_bar.showMessage("Processing... t0 correction will take a minute or two. Be patient.")
        self.thread = QThread()              # Create a QThread object
        self.worker = pyqt5_thread_worker_t0_correction(parent_pyqt5_QMainWindow=self)    # Create the Thread Worker
        self.worker.moveToThread(self.thread) # Move the worker to the thread
        self.thread.started.connect(self.worker.run)        # Connect Signals: When thread starts -> Run worker logic
        self.worker.finished.connect(self.thread.quit)          # Connect Signals: When worker finishes -> Quit thread, Clean up objects
        self.worker.finished.connect(self.worker.deleteLater)   # Connect Signals: When worker finishes -> Quit thread, Clean up objects
        self.thread.finished.connect(self.thread.deleteLater)   # Connect Signals: When worker finishes -> Quit thread, Clean up objects
        self.worker.result_ready.connect(self.on_t0_correction_complete)   # Connect Signals: When data is ready -> Update the GUI
        self.worker.error_occurred.connect(lambda e: self.status_bar.showMessage(f"Error: {e}"))    # Connect Signals: if error occurs -> Show error message        
        self.btn_correct.setEnabled(False); self.btn_save_matrix.setEnabled(False); self.btn_add_t0.setEnabled(False); self.btn_fit_t0.setEnabled(False)  # Disable the buttons so they don't click it twice 
        self.thread.start()             # Start the thread


    def on_t0_correction_complete(self, corrected_data):
        # Destroy the fit line and reset click_history
        self.click_history = [[-10000,-10000]]    # start with a value of zero in there to allow yourself to check if values are increasing, then cut away this zero for final fitting
        self.t0_points_scatter.set_offsets(np.array(self.click_history))
        self.t0_fit_line.set_visible(False)
        self.t0_fit_line.set_data([],[])
        
        # Count NaNs, report them, fill them
        nan_count = np.sum(np.isnan(corrected_data))
        if nan_count > 0:
            print(f"{nan_count} NaNs found and set to 0.")
            corrected_data[np.isnan(corrected_data)] = 0

        self.TA_data = corrected_data   # Update the TA_matrix Data
        self.pcolormesh_transects.TA_image_axis_han.collections[0].set_array(self.TA_data.ravel())      # Update the TA_matrix plot data
        self.TA_matrix_interpolator = RegularGridInterpolator((self.TA_matrix_delay_times, np.flip(self.TA_matrix_wavelengths)), np.fliplr(self.TA_data), bounds_error=False, fill_value=None) # Recreate the interpolator
        self.pcolormesh_transects.update_transects()
        self.canvas.draw()
        
        # Re-enable buttons
        self.btn_correct.setEnabled(True); self.btn_save_matrix.setEnabled(True); self.btn_add_t0.setEnabled(True); self.btn_fit_t0.setEnabled(True)  # Re-allow these buttons
        self.status_bar.showMessage("Correction Complete via Background Thread.")



    def save_TA_matrix(self):
        """Saves the current (corrected) TA matrix to disk."""
        if isinstance(self.TA_matrix_input, str):
            filename = f"{self.TA_matrix_input}.t0_corr.csv"
        else:
            filename = "output.t0_corr.csv"

        # Reconstruct the full matrix structure (Wavelengths in row 0, Delays in col 0)
        output_matrix = np.zeros(self.TA_matrix.shape)
        output_matrix[0, 1:] = self.TA_matrix_wavelengths
        output_matrix[1:, 0] = self.TA_matrix_delay_times
        output_matrix[1:, 1:] = self.TA_data

        try:
            np.savetxt(filename, output_matrix, delimiter=',')
            self.status_bar.showMessage(f"Saved Matrix: {filename}")
        except Exception as e:
            self.status_bar.showMessage(f"Save Failed: {e}")






class pyqt5_thread_worker_t0_correction(QThread):
    finished = pyqtSignal()                      # define a signal: Task is done
    result_ready = pyqtSignal(object)            # define a signal: Here is the data (sends a numpy array)
    error_occurred = pyqtSignal(str)             # define a signal: Something went wrong

    def __init__(self, parent_pyqt5_QMainWindow):
        super().__init__()
        self.parent_pyqt5_QMainWindow = parent_pyqt5_QMainWindow

    def run(self):
        try:
            # Perform heavy calculation here: t0 CHIRP CORRECTION (Interpolation)
            delay_times = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times
            wavelengths = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths
            t0_each_wavelength = self.parent_pyqt5_QMainWindow.t0_each_wavelength
            # Create grid of original data points
            grid_delay_times = np.outer(delay_times, np.ones(len(wavelengths)))      #outer product to make a grid
            grid_wavelengths = np.outer(np.ones(len(delay_times)), wavelengths)      #outer product to make a grid
            # Pad the TA_data to avoid NaNs at edges during interpolation
            TA_data_padded = np.vstack((self.parent_pyqt5_QMainWindow.TA_data[0, :], self.parent_pyqt5_QMainWindow.TA_data[0, :], self.parent_pyqt5_QMainWindow.TA_data[0, :], self.parent_pyqt5_QMainWindow.TA_data))
            TA_data_padded = np.vstack((TA_data_padded, TA_data_padded[-1, :], TA_data_padded[-1, :], TA_data_padded[-1, :]))
            top_diff = grid_delay_times[0, :] - grid_delay_times[1, :]; bottom_diff = grid_delay_times[-1, :] - grid_delay_times[-2, :] 
            grid_delay_times_padded = np.vstack((grid_delay_times[0, :]+top_diff*3, grid_delay_times[0, :]+top_diff*2, grid_delay_times[0, :]+top_diff, grid_delay_times ) )
            grid_delay_times_padded = np.vstack((grid_delay_times_padded, grid_delay_times_padded[-1, :]+bottom_diff, grid_delay_times_padded[-1, :]+bottom_diff*2, grid_delay_times_padded[-1, :]+bottom_diff*3 ))
            grid_wavelengths_padded = np.vstack((grid_wavelengths[0, :], grid_wavelengths[0, :], grid_wavelengths[0, :], grid_wavelengths))
            grid_wavelengths_padded = np.vstack((grid_wavelengths_padded, grid_wavelengths_padded[-1, :], grid_wavelengths_padded[-1, :], grid_wavelengths_padded[-1, :]))
            t0_corrected_array_delay_times_padded = grid_delay_times_padded - t0_each_wavelength
            # Perform the interpolation
            corrected_data = griddata( (t0_corrected_array_delay_times_padded.ravel(), grid_wavelengths_padded.ravel()), TA_data_padded.ravel(), (grid_delay_times, grid_wavelengths), method='linear')
            # BACKGROUND REMOVAL
            background_each_wavelength = np.zeros(len(wavelengths))
            t0_idx = int(np.abs(delay_times).argmin()) # Index of time zero
            # Logic to determine start index for background averaging
            st_idx = int(delay_times.argmin()) # Find which ench of the TA matrix holds the smallest delay times
            # Iterative background removal
            for repeat in range(0, 30):
                for j in range(1, len(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths) - 2):
                    if st_idx < t0_idx:
                        background_each_wavelength[j] = np.sum(corrected_data[st_idx:t0_idx-15, j-1:j+1]) / (t0_idx - st_idx) / 3
                        background_each_wavelength[0] = np.sum(corrected_data[st_idx:t0_idx-15, 0]) / (t0_idx - st_idx)
                        background_each_wavelength[-1] = np.sum(corrected_data[st_idx:t0_idx-15, -1]) / (t0_idx - st_idx)
                    elif st_idx > t0_idx:
                        background_each_wavelength[j] = np.sum(corrected_data[t0_idx:st_idx+15, j-1:j+1]) / (st_idx - t0_idx) / 3
                        background_each_wavelength[0] = np.sum(corrected_data[t0_idx:st_idx+15, 0]) / (st_idx - t0_idx)
                        background_each_wavelength[-1] = np.sum(corrected_data[t0_idx:st_idx+15, -1]) / (st_idx - t0_idx)
                corrected_data[:] = corrected_data - background_each_wavelength


            self.result_ready.emit(corrected_data)  # FINISHED: Send the corrected data back to the main thread
            self.finished.emit()                     # # FINISHED: Let the thread know we're done

        except Exception as e:
            self.error_occurred.emit(str(e))






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
        TA_matrix_input = sys.argv[1]
    else:
        # Fallback default if no file is provided
        TA_matrix_input = "HHHF_Zn_heme_ZnCl_p425nm_red_300uW.h5"  # default input for testing:
    # Start the application
    window = TA_t0_correction_and_background_removal(TA_matrix_input) #The QApplication instance must be created before this.  The init method recognizes QApplication already exists and uses it.
    
