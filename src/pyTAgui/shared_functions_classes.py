#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import os
import time
import h5py
import numpy as np
from scipy.ndimage import generic_filter

# Matplotlib Imports (The Plotting Library)
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# PyQt5 Imports (The GUI Library)
from PyQt5.QtWidgets import (QGridLayout, QLineEdit, QLabel, QPushButton, QGroupBox)
from PyQt5.QtCore import QTimer, Qt

class embedded_matplotlib_pcolormesh_transects():
    def __init__(self, parent_pyqt5_QMainWindow, frac_horiz=1.0, frac_vert=1.0):
        self.parent_pyqt5_QMainWindow = parent_pyqt5_QMainWindow

        scale_axes = np.array([frac_horiz, frac_vert, frac_horiz, frac_vert])  #This is so the caller can make the pcolormesh and transect axes smaller as needed.

        ### Create Axes (Using your exact coordinates) 
        main_TA_image_axes = np.array([0.06, 1-(1-0.31)*frac_vert, 0.66, 0.685*frac_vert])    #[left   bottom    fraction_horizontal   fraction_vertical ]
        horiz_transect_axes = np.array([0.06, 1-(1-0.05)*frac_vert, 0.66, 0.20*frac_vert])    #[left   bottom    fraction_horizontal   fraction_vertical ]
        vert_transect_axes = np.array([0.79, 1-(1-0.31)*frac_vert, 0.2, 0.685*frac_vert])     #[left   bottom    fraction_horizontal   fraction_vertical ]
        colorbar_axes      = np.array([0.75, 1-(1-0.22)*frac_vert, 0.23, 0.03*frac_vert])     #[left   bottom    fraction_horizontal   fraction_vertical ]

        # Main TA Image
        self.TA_image_axis_han = self.parent_pyqt5_QMainWindow.fig_han.add_axes(main_TA_image_axes )    #[left bottom fraction_horizontal  fraction_vertical ]
        self.TA_image_axis_han.set_ylabel('delay time / ps')
        self.TA_image_axis_han.set_xlabel('wavelength / nm')

        # Horizontal Transect (Bottom)
        self.ax_horiz = self.parent_pyqt5_QMainWindow.fig_han.add_axes(horiz_transect_axes, sharex=self.TA_image_axis_han)  #[left bottom fraction_horizontal  fraction_vertical ]
        self.ax_horiz.grid(True, linestyle=':')
        self.ax_horiz.set_xlabel('wavelength / nm')
        
        # Vertical Transect (Right)
        self.ax_vert = self.parent_pyqt5_QMainWindow.fig_han.add_axes(vert_transect_axes, sharey=self.TA_image_axis_han)  #[left bottom fraction_horizontal  fraction_vertical ]
        self.ax_vert.set_ylabel('delay time / ps', labelpad=2)
        self.ax_vert.grid(True, linestyle=':')
        self.ax_vert.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        # Colorbar Axis
        self.axis_colorbar = self.parent_pyqt5_QMainWindow.fig_han.add_axes(colorbar_axes)   #[left bottom fraction_horizontal  fraction_vertical ]

        # Initial Plotting of the TA Image
        min_max = [np.percentile(self.parent_pyqt5_QMainWindow.TA_data.flatten(),2), np.percentile(self.parent_pyqt5_QMainWindow.TA_data.flatten(),98)]
        self.TA_colormap = create_TA_Blue_White_Red_Black_colormap(min_max)
        # Plot the TA data as a pcolormesh (like imshow but with correct axis scaling)
        self.pclrmsh = self.TA_image_axis_han.pcolormesh(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths, self.parent_pyqt5_QMainWindow.TA_matrix_delay_times, self.parent_pyqt5_QMainWindow.TA_data, vmin=min_max[0], vmax=min_max[1], cmap=self.TA_colormap, rasterized=True)
        self.home_xlim = self.TA_image_axis_han.get_xlim()
        self.home_ylim = self.TA_image_axis_han.get_ylim()
        # Colormap and colorbar setup
        self.cbar = self.parent_pyqt5_QMainWindow.fig_han.colorbar(self.pclrmsh, cax=self.axis_colorbar, orientation='horizontal')
        self.cbar.ax.tick_params(axis='x',  rotation=0)
        self.cbar.ax.ticklabel_format(style='sci', axis='x', scilimits=(0.1, 10))
        self.cbar.set_label(r'-$\Delta$T / T')

        # Crosshairs (Initialized at index)
        self.crosshair_v_idx = self.parent_pyqt5_QMainWindow.TA_data_cols // 2
        self.crosshair_h_idx = self.parent_pyqt5_QMainWindow.TA_data_rows - 5
        self.vert_cut_num_ave = 1
        self.horiz_cut_num_ave = 1
        self.crosshair_v = self.TA_image_axis_han.axvline(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx], color='k', linestyle='--', alpha=0.2)
        self.crosshair_h = self.TA_image_axis_han.axhline(self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx], color='k', linestyle='--', alpha=0.2)

        # Transect Lines (Empty/Initial)
        self.line_horiz, = self.ax_horiz.plot(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths, self.parent_pyqt5_QMainWindow.TA_data[self.crosshair_h_idx,:], color='k', linewidth=1)
        self.red_dot_horiz = self.ax_horiz.scatter(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx], self.parent_pyqt5_QMainWindow.TA_data[self.crosshair_h_idx, self.crosshair_v_idx], color='red', s=40, zorder=2)
        self.line_vert, = self.ax_vert.plot(self.parent_pyqt5_QMainWindow.TA_data[:,self.crosshair_v_idx], self.parent_pyqt5_QMainWindow.TA_matrix_delay_times, color='k', linewidth=1)
        self.red_dot_vert = self.ax_vert.scatter(self.parent_pyqt5_QMainWindow.TA_data[self.crosshair_h_idx, self.crosshair_v_idx], self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx], color='red', s=40, zorder=2)

        # create pyqt5 control 1: Axis Limits 
        self.linlog_threshold_time_axis = 0.01  # ps
        grp_ax_limits = QGroupBox("TA Plot Axis Limits")
        grp_ax_limits.setStyleSheet('QLabel { font-size: 11pt; } QLineEdit { font-size: 11pt; }')
        grid_ax_limits = QGridLayout()
        grid_ax_limits.setVerticalSpacing(2)
        grid_ax_limits.setContentsMargins(2, 5, 2, 2)                   # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        # Wavelength Axis Limits
        grid_ax_limits.addWidget(QLabel("Wavelength Max:"), 0, 0)       # row ,  column)
        self.txt_w_max = QLineEdit(f"{np.max(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths):4.2f}")
        self.txt_w_max.returnPressed.connect(self.update_axis_limits)
        grid_ax_limits.addWidget(self.txt_w_max, 0, 1)
        grid_ax_limits.addWidget(QLabel("Wavelength Min:"), 1, 0)
        self.txt_w_min = QLineEdit(f"{np.min(self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths):4.2f}")
        self.txt_w_min.returnPressed.connect(self.update_axis_limits)
        grid_ax_limits.addWidget(self.txt_w_min, 1, 1)
        # Time Axis Limits
        grid_ax_limits.addWidget(QLabel("Time Max:"), 2, 0)
        self.txt_t_max = QLineEdit(f"{np.max(self.parent_pyqt5_QMainWindow.TA_matrix_delay_times):5.2f}")
        self.txt_t_max.returnPressed.connect(self.update_axis_limits)
        grid_ax_limits.addWidget(self.txt_t_max, 2, 1)
        grid_ax_limits.addWidget(QLabel("Time Min:"), 3, 0)
        self.txt_t_min = QLineEdit(f"{np.min(self.parent_pyqt5_QMainWindow.TA_matrix_delay_times):5.2f}")
        self.txt_t_min.returnPressed.connect(self.update_axis_limits)
        grid_ax_limits.addWidget(self.txt_t_min, 3, 1)
        grp_ax_limits.setLayout(grid_ax_limits)
        self.parent_pyqt5_QMainWindow.control_panel_layout.addWidget(grp_ax_limits)


        # create pyqt5 control 2: Color Scale
        grp_colormap = QGroupBox("Color Map Scaling")
        grp_colormap.setStyleSheet('QLabel { font-size: 11pt; } QLineEdit { font-size: 11pt; }')
        grid_clrmap_limits = QGridLayout()   
        grid_clrmap_limits.setVerticalSpacing(2)
        grid_clrmap_limits.setContentsMargins(2, 5, 2, 2)                      # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        grid_clrmap_limits.addWidget(QLabel("Colormap max:"), 0, 0)            # row ,  column)
        clim = self.pclrmsh.get_clim()
        self.txt_cmap_max = QLineEdit( f"{clim[1]:.5f}" )
        self.txt_cmap_max.returnPressed.connect(self.update_colormap_manual)
        grid_clrmap_limits.addWidget(self.txt_cmap_max, 0, 1)
        grid_clrmap_limits.addWidget(QLabel("Colormap min:"), 1, 0)
        clim = self.pclrmsh.get_clim()
        self.txt_cmap_min = QLineEdit( f"{clim[0]:.5f}" )
        self.txt_cmap_min.returnPressed.connect(self.update_colormap_manual)
        grid_clrmap_limits.addWidget(self.txt_cmap_min, 1, 1)     
        grp_colormap.setLayout(grid_clrmap_limits)
        self.parent_pyqt5_QMainWindow.control_panel_layout.addWidget(grp_colormap)


        # create pyqt5 control 3: Time Axis Lin-Log Scaling
        self.linear_threshold_time_axis = 0.01
        grp_scale = QGroupBox("Time Axis Scaling")
        grp_scale.setStyleSheet('QLabel { font-size: 11pt; } QLineEdit { font-size: 11pt; } QPushButton {font-size: 11pt; }')
        grid_linlog = QGridLayout()
        grid_linlog.setVerticalSpacing(2)
        grid_linlog.setContentsMargins(2, 5, 2, 2)                      # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        grid_linlog.addWidget(QLabel("Lin-Log limit (ps):"), 0, 0)
        self.txt_linlog_thresh = QLineEdit(str(self.linlog_threshold_time_axis))
        self.txt_linlog_thresh.returnPressed.connect(self.update_linlog_thresh)
        grid_linlog.addWidget(self.txt_linlog_thresh, 0, 1)
        self.btn_linlog_time = QPushButton("Lin-Log y-axis")
        self.btn_linlog_time.setCheckable(True)
        self.btn_linlog_time.setStyleSheet("QPushButton { background-color: #F8F8F8; border: 1px solid #ADADAD; border-radius: 4px; padding: 3px;}  QPushButton:checked {font-size: 11pt; background-color: #E0E0E0; border: 1px solid #ADADAD;}")
        self.btn_linlog_time.clicked.connect(self.on_linlog_toggle)
        grid_linlog.addWidget(self.btn_linlog_time, 1, 0, 1, 2)         # row,   column wthin row,   # of rows to occupy,    # of columns to occupy)
        grp_scale.setLayout(grid_linlog)
        self.parent_pyqt5_QMainWindow.control_panel_layout.addWidget(grp_scale)


        # create pyqt5 control 4: Averaging
        grp_avg = QGroupBox("Transect Averaging")
        grp_avg.setStyleSheet('QLabel { font-size: 11pt; } QLineEdit { font-size: 11pt; }')
        grid_avg = QGridLayout()
        grid_avg.setVerticalSpacing(2)
        grid_avg.setContentsMargins(2, 5, 2, 2)
        grid_avg.addWidget(QLabel("Horiz Avg (rows):"), 0, 0)
        self.txt_h_avg = QLineEdit("1")
        self.txt_h_avg.returnPressed.connect(self.update_transects)
        grid_avg.addWidget(self.txt_h_avg, 0, 1)
        grid_avg.addWidget(QLabel("Vert Avg (cols):"), 1, 0)
        self.txt_v_avg = QLineEdit("1")
        self.txt_v_avg.returnPressed.connect(self.update_transects)
        grid_avg.addWidget(self.txt_v_avg, 1, 1)
        grp_avg.setLayout(grid_avg)
        self.parent_pyqt5_QMainWindow.control_panel_layout.addWidget(grp_avg)

        # Set custom formatter for the toolbar status message
        self.TA_image_axis_han.format_coord = self.format_toolbar_status_msg_TA_matrix
        self.ax_horiz.format_coord = self.format_toolbar_status_msg_horiz_transect
        self.ax_vert.format_coord = self.format_toolbar_status_msg_vert_transect

        # # Setup the QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_every_500ms) # Connect the 'timeout' signal to your function
        self.timer.start(500)  # Start the timer (Interval in milliseconds)


    #########################
    ####### Update every 500 ms
    def run_every_500ms(self):
        #This function gets called automatically every 500ms.
        xlim = self.TA_image_axis_han.get_xlim()
        ylim = self.TA_image_axis_han.get_ylim()
        try:
            if np.abs(xlim[0] - float(self.txt_w_min.text())) > 0.1 or np.abs(xlim[1] - float(self.txt_w_max.text())) > 0.1 or np.abs(ylim[0] - float(self.txt_t_min.text())) > 0.1 or np.abs(ylim[1] - float(self.txt_t_max.text())) > 0.1:
                if self.parent_pyqt5_QMainWindow.focusWidget() != self.txt_w_min and self.parent_pyqt5_QMainWindow.focusWidget() != self.txt_w_max and self.parent_pyqt5_QMainWindow.focusWidget() != self.txt_t_min and self.parent_pyqt5_QMainWindow.focusWidget() != self.txt_t_max:
                    self.txt_w_min.setText(f"{xlim[0]:.2f}")
                    self.txt_w_max.setText(f"{xlim[1]:.2f}")
                    self.txt_t_min.setText(f"{ylim[0]:.2f}")
                    self.txt_t_max.setText(f"{ylim[1]:.2f}")
                    self.update_transects()
        except ValueError:
            pass



    ##############################################
    ####### Callbacks for Interactivity 

    def update_transects(self):
        ### Updates the side plots based on current crosshair positions and averaging settings.        
        try:
            self.horiz_cut_num_ave = int(self.txt_h_avg.text())
            self.vert_cut_num_ave = int(self.txt_v_avg.text())
        except ValueError:
            self.parent_pyqt5_QMainWindow.status_bar.showMessage("Invalid Transect Input")
        # Determine Padding for Averaging, with safe bounds checking
        pad_h = 0
        if self.horiz_cut_num_ave > 1:
            pad_h = int((self.horiz_cut_num_ave - 1) // 2) # simplified logic
        pad_v = 0
        if self.vert_cut_num_ave > 1:
            pad_v = int((self.vert_cut_num_ave - 1) // 2)
        # Safe bounds check
        h_start = max(0, self.crosshair_h_idx - pad_h)
        h_end = min(self.parent_pyqt5_QMainWindow.TA_data_rows, self.crosshair_h_idx + pad_h + 1)
        v_start = max(0, self.crosshair_v_idx - pad_v)
        v_end = min(self.parent_pyqt5_QMainWindow.TA_data_cols, self.crosshair_v_idx + pad_v + 1)

        # Find current zoom xlim and ylim so you can automatically rescale transects
        xlim = self.TA_image_axis_han.get_xlim()
        w_mask = (self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths >= xlim[0]) & (self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths <= xlim[1])
        ylim = self.TA_image_axis_han.get_ylim()
        t_mask = (self.parent_pyqt5_QMainWindow.TA_matrix_delay_times >= ylim[0]) & (self.parent_pyqt5_QMainWindow.TA_matrix_delay_times <= ylim[1])

        # Horizontal Transect (Slicing the matrix)
        # Average the rows around the crosshair
        averaged_spectrum = np.mean(self.parent_pyqt5_QMainWindow.TA_data[h_start:h_end, :], axis=0)
        self.line_horiz.set_ydata(averaged_spectrum)
        # Rescale view
        self.ax_horiz.set_ylim([averaged_spectrum[w_mask].min(), averaged_spectrum[w_mask].max()])
        # Update red dot
        self.red_dot_horiz.set_offsets([self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx], averaged_spectrum[self.crosshair_v_idx]])

        # Vertical Transect
        # Average the columns around the crosshair
        averaged_transient = np.mean(self.parent_pyqt5_QMainWindow.TA_data[:, v_start:v_end], axis=1)
        self.line_vert.set_xdata(averaged_transient)
        # Rescale view
        self.ax_vert.set_xlim([averaged_transient[t_mask].min(), averaged_transient[t_mask].max()])
        # Update red dot
        self.red_dot_vert.set_offsets([averaged_transient[self.crosshair_h_idx], self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]])

        self.parent_pyqt5_QMainWindow.canvas.draw()


    def update_crosshair_lines(self):
        ### Updates just the dashed lines on the heatmap.
        # Get actual float values
        w_val = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx]
        t_val = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]
        self.crosshair_v.set_xdata([w_val, w_val])
        self.crosshair_h.set_ydata([t_val, t_val])
        self.parent_pyqt5_QMainWindow.canvas.draw()
    

    def update_axis_limits(self):
        try:
            self.allow_TA_matrix_axis_limit_update = False
            self.TA_image_axis_han.set_xlim(float(self.txt_w_min.text()), float(self.txt_w_max.text()))
            self.TA_image_axis_han.set_ylim(float(self.txt_t_min.text()), float(self.txt_t_max.text()))
            self.parent_pyqt5_QMainWindow.canvas.draw()
            if self.parent_pyqt5_QMainWindow.status_bar.currentMessage()=="Invalid Axis Limit Input":
                self.parent_pyqt5_QMainWindow.status_bar.showMessage("")
        except ValueError:
            self.allow_TA_matrix_axis_limit_update = True
            self.parent_pyqt5_QMainWindow.status_bar.showMessage("Invalid Axis Limit Input")


    def update_colormap_manual(self):
        try:
            self.TA_colormap = create_TA_Blue_White_Red_Black_colormap([float(self.txt_cmap_min.text()), float(self.txt_cmap_max.text())])
            self.pclrmsh.set_cmap(self.TA_colormap)
            self.pclrmsh.set_clim(float(self.txt_cmap_min.text()), float(self.txt_cmap_max.text()))
            self.parent_pyqt5_QMainWindow.canvas.draw()
            if self.parent_pyqt5_QMainWindow.status_bar.currentMessage()=="Invalid Colormap Input":
                self.parent_pyqt5_QMainWindow.status_bar.showMessage("")
        except:
            self.parent_pyqt5_QMainWindow.status_bar.showMessage("Invalid Colormap Input")
        

    def on_linlog_toggle(self, checked):
        try:
            self.linlog_threshold_time_axis = float(self.txt_linlog_thresh.text())
        except ValueError:
            self.parent_pyqt5_QMainWindow.status_bar.showMessage("Invalid LinLog Threshold Input")
            self.btn_linlog_time.setChecked(False)
            return
        self.parent_pyqt5_QMainWindow.status_bar.showMessage("")
        if checked:
            self.TA_image_axis_han.set_yscale('symlog', linthresh=self.linlog_threshold_time_axis)
        else:
            self.TA_image_axis_han.set_yscale('linear')
        self.parent_pyqt5_QMainWindow.canvas.draw()



    def update_linlog_thresh(self):
        try:
            self.allow_TA_matrix_axis_limit_update = False
            self.linlog_threshold_time_axis = float(self.txt_linlog_thresh.text())
            # If currently in linlog mode, update immediately
            if self.btn_linlog_time.isChecked():
                self.TA_image_axis_han.set_yscale('symlog', linthresh=self.linlog_threshold_time_axis)
                self.parent_pyqt5_QMainWindow.canvas.draw()
            if self.parent_pyqt5_QMainWindow.status_bar.currentMessage()=="Invalid LinLog Threshold Input":
                self.parent_pyqt5_QMainWindow.status_bar.showMessage("")
            self.allow_TA_matrix_axis_limit_update = True
        except ValueError:
            self.allow_TA_matrix_axis_limit_update = True
            self.parent_pyqt5_QMainWindow.status_bar.showMessage("Invalid LinLog Threshold Input")



    def format_toolbar_status_msg_TA_matrix(self, w_mouse, t_mouse):
        # Custom formatter for the toolbar message. Matplotlib calls this automatically with x, y coords.
        try:
            z_mouse = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_mouse, w_mouse])[0]    # Look up Z-value
            w_mouse_str = f"{w_mouse:>6.1f}"    # convert to string
            t_mouse_str = f"{t_mouse:>6.1f}"    # convert to string
            z_mouse_str = f"{z_mouse:>+3.8f}"   # convert to string
            w_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx]
            t_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]
            z_val_crosshairs = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_crosshair, w_crosshair])[0]
            w_crosshair_str = f"{w_crosshair:>5.1f}"
            t_crosshair_str = f"{t_crosshair:>6.1f}"
            z_crosshair_str = f"{z_val_crosshairs:>+3.8f}"
        except Exception:
            return ""
        msg = f"Mouse: w={w_mouse_str} nm t={t_mouse_str} ps z={z_mouse_str}\nCrosshairs: w={w_crosshair_str}nm, t={t_crosshair_str}ps, z={z_crosshair_str}"
        return msg

    def format_toolbar_status_msg_horiz_transect(self, w_mouse, z_mouse):
        # Custom formatter for the toolbar message. Matplotlib calls this automatically with x, y coords.
        try:
            t_mouse = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]
            w_mouse_str = f"{w_mouse:>6.1f}"    # convert to string
            t_mouse_str = f"{t_mouse:>6.1f}"    # convert to string
            z_mouse_on_transect = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_mouse, w_mouse])[0]
            z_mouse_str = f"{z_mouse_on_transect:>+3.8f}"   # convert to string
            w_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx]
            t_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]
            z_val_crosshairs = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_crosshair, w_crosshair])[0]
            w_crosshair_str = f"{w_crosshair:>5.1f}"
            t_crosshair_str = f"{t_crosshair:>6.1f}"
            z_crosshair_str = f"{z_val_crosshairs:>+3.8f}"
        except Exception:
            return ""
        msg = f"Mouse: w={w_mouse_str} nm t={t_mouse_str} ps z={z_mouse_str}\nCrosshairs: w={w_crosshair_str}nm, t={t_crosshair_str}ps, z={z_crosshair_str}"
        return msg
    
    def format_toolbar_status_msg_vert_transect(self, z_mouse, t_mouse):
        # Custom formatter for the toolbar message. Matplotlib calls this automatically with x, y coords.
        try:
            w_mouse = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx]
            w_mouse_str = f"{w_mouse:>6.1f}"    # convert to string
            t_mouse_str = f"{t_mouse:>6.1f}"    # convert to string
            z_mouse_on_transect = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_mouse, w_mouse])[0]
            z_mouse_str = f"{z_mouse_on_transect:>+3.8f}"   # convert to string
            w_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_wavelengths[self.crosshair_v_idx]
            t_crosshair = self.parent_pyqt5_QMainWindow.TA_matrix_delay_times[self.crosshair_h_idx]
            z_val_crosshairs = self.parent_pyqt5_QMainWindow.TA_matrix_interpolator([t_crosshair, w_crosshair])[0]
            w_crosshair_str = f"{w_crosshair:>5.1f}"
            t_crosshair_str = f"{t_crosshair:>6.1f}"
            z_crosshair_str = f"{z_val_crosshairs:>+3.8f}"
        except Exception:
            return ""
        msg = f"Mouse: w={w_mouse_str} nm t={t_mouse_str} ps z={z_mouse_str}\nCrosshairs: w={w_crosshair_str}nm, t={t_crosshair_str}ps, z={z_crosshair_str}"
        return msg
























def plot_fractional_change_TA_probe_counts(TA_filename):
    sweep_0_probe_counts = load_hdf5_data(TA_filename,'Spectra/Sweep_0_Probe_Spectrum')
    sweep_1_probe_counts = load_hdf5_data(TA_filename,'Spectra/Sweep_1_Probe_Spectrum')
    wavelengths = get_TA_matrix(TA_filename)[0,1:]
    fractional_change = (sweep_1_probe_counts - sweep_0_probe_counts) / sweep_0_probe_counts
    fig_han = plt.figure()
    ax=fig_han.add_axes([0.135,0.15,0.75,0.75])
    ax.plot(wavelengths, fractional_change, 'b', label='fractional change \n probe counts')
    ax2 = ax.twinx()
    ax2.plot(wavelengths, sweep_0_probe_counts, 'g', label = 'probe counts')
    ax.set_xlim(450,800)
    ax.set_ylim(-0.1,0.1)
    ax2.set_ylim(0,3000)
    ax.set_xlabel('wavelength / nm')
    ax.set_ylabel('fractional change', color = 'b')
    ax.tick_params(axis='y', colors='b')
    ax2.set_ylabel('probe counts', color='g')
    ax2.tick_params(axis='y', colors='g')
    ax.set_title(TA_filename)
    plt.show()
    





def TA_matrix_window_average(TA_matrix,window_size):
    TA_matrix = get_TA_matrix(TA_matrix)
    TA_matrix_window_averaged = TA_matrix.copy()
    TA_matrix_window_averaged[1:,1:] = generic_filter(TA_matrix[1:,1:], np.mean, size=window_size, mode='constant', cval=0)
    return TA_matrix_window_averaged



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
    if isinstance(min_max[0], str):
        min = float(min_max[0])
        max = float(min_max[1])
    else:
        min = min_max[0]
        max = min_max[1]
    #white_fractional_location is a number between 0 and 1
    TA_colormap = np.ones([100,4])
    white_fractional_location = (0.0 - min)/ (max - min)
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
    if isinstance(TA_matrix, np.ndarray):
        return TA_matrix
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






def list_hdf5_contents(filename):
    # Opens an HDF5 file and prints its hierarchy (Groups/Datasets) and any attributes (variables) associated with them.

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found in the current directory.")
        return

    def print_attrs(name, obj):
        # Helper to print attributes (variables) of an object
        if obj.attrs:
            print(f"{'  ' * (indent + 1)}Attributes:")
            for key, val in obj.attrs.items():
                print(f"{'  ' * (indent + 2)}- {key}: {val}")

    def recurse(name, node, indent=0):
        # Determine if it's a Group or Dataset
        if isinstance(node, h5py.Group):
            print(f"{'  ' * indent}📂 Group: {name}")
            
            # Print attributes of the group
            if node.attrs:
                print(f"{'  ' * (indent + 1)}Attributes:")
                for key, val in node.attrs.items():
                    print(f"{'  ' * (indent + 2)}- {key}: {val}")
            
            # Recurse into children
            for child_name, child_node in node.items():
                recurse(child_name, child_node, indent + 1)
                
        elif isinstance(node, h5py.Dataset):
            # Print dataset info (shape and dtype are usually helpful context)
            print(f"{'  ' * indent}📄 Dataset: {name} (Shape: {node.shape}, Type: {node.dtype})")
            
            # Print attributes of the dataset
            if node.attrs:
                print(f"{'  ' * (indent + 1)}Attributes:")
                for key, val in node.attrs.items():
                    print(f"{'  ' * (indent + 2)}- {key}: {val}")

    try:
        with h5py.File(filename, 'r') as f:
            print(f"--- Structure of {filename} ---")
            # Start recursion from the root group
            # We manually iterate root keys to keep the indent logic clean
            if f.attrs:
                print("Root Attributes:")
                for key, val in f.attrs.items():
                    print(f"  - {key}: {val}")
            indent = 0        
            for key, val in f.items():
                recurse(key, val, indent)
                
    except OSError:
        print(f"Error: Could not open '{filename}'. Is it a valid HDF5 file?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

