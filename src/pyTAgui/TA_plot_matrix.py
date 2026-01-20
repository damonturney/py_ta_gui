#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# PyQt5 Imports (The GUI Library)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QLabel, QPushButton, QStatusBar, QGroupBox)
from PyQt5.QtCore import QTimer, Qt

# Matplotlib Imports (The Plotting Library)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT 
from matplotlib.figure import Figure

# Import the shared functions and classes
import src.pyTAgui.shared_functions_classes as TA_sh


class TA_plot_matrix_app():   
    def __init__(self, TA_matrix_input):
        self.app = QApplication(sys.argv)                                           # Create the QApplication instance           
        self.TA_matrix_input = TA_matrix_input
        self.main_pyqt5_window = TA_plot_matrix_main_window(self.TA_matrix_input)   # Create the main pyqt5 window
        self.main_pyqt5_window.show()                                               # Display the window
        sys.exit(self.app.exec())                                                   # Start the event loop        



class TA_plot_matrix_main_window(QMainWindow):    #This nomenclature causes this class to inherit all the methods and properties of the QMainWindow class from PyQT5.
    def __init__(self, TA_matrix_input):
        
        super().__init__() #This executes a lot of code that builds the PyQT5 window using the QMainWindow class.
        
        # pyqt5 window setup: needs the QMainWindow object to be inherited into this class. 
        self.setWindowTitle("TA Matrix Viewer")
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

        # create a pyqt5 button control: Save Transient
        self.btn_save = QPushButton("Save Plotted Transient (.csv)")

        self.btn_save.clicked.connect(self.save_TA_transient)
        self.control_panel_layout.addWidget(self.btn_save)
        self.btn_save.setStyleSheet('QPushButton { font-size: 11pt; }')
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

    def save_TA_transient(self):
        w_val = self.TA_matrix_wavelengths[self.pcolormesh_transects.crosshair_v_idx]
        if isinstance(self.TA_matrix_input, str):
            filename = f"{self.TA_matrix_input}.transient{w_val:.1f}nm.csv"
        else:
            filename = f"transient{w_val:.1f}nm.csv"
        data = np.column_stack((self.pcolormesh_transects.line_vert.get_xdata(), self.pcolormesh_transects.line_vert.get_ydata()))
        np.savetxt(filename, data, delimiter=',')
        self.status_bar.showMessage(f"Saved: {filename}")


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
    window = TA_plot_matrix_app(TA_matrix_input) #The QApplication instance must be created before this.  The init method recognizes QApplication already exists and uses it.
    
