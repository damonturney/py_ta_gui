#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

# Import our custom codes
import shared_supporting_functions as TA_sh

plt.rcParams['font.family'] = 'monospace'

plt.close('all')

# This function requires TA_matrix to hold a row vector of wavelengths in row 1, and a vector of probe delay times in column 1.
def plot_TA_matrix(TA_matrix_input):

    # Get the TA_matrix data
    TA_matrix = TA_sh.get_TA_matrix(TA_matrix_input)

    # Extract the wavelengths and time delays
    TA_matrix_wavelengths = TA_matrix[0,1:]
    TA_matrix_delay_times = TA_matrix[1:,0]
    # Crop the TA image down to remove the wavelengths and delay times
    TA_data = TA_matrix[1:,1:]


    # Create a good color map
    min_max = [np.percentile(TA_data.flatten(),2), np.percentile(TA_data.flatten(),98)]
    TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )


    # Create figure
    fig_han = plt.figure(figsize=(12, 8))                        #fig_han.get_size_inches()
    fig_han.tight_layout()

    # Plot up the main TA matrix image and colorbar
    TA_image_axis_han = fig_han.add_axes([0.06,0.31,0.5,0.68])            # [left bottom fraction_horizontal  fraction_vertical ]
    pclrmsh = TA_image_axis_han.pcolormesh(TA_matrix_wavelengths, TA_matrix_delay_times, TA_data, vmin=min_max[0], vmax=min_max[1], cmap=TA_colormap, mouseover=True)
    TA_image_axis_han.axes.yaxis.set_label_text('delay time / ps')
    TA_image_axis_han.axes.xaxis.set_label_text('wavelength / nm')
    axis_colorbar = fig_han.add_axes([0.87, 0.48, 0.02, 0.5])
    cbar = fig_han.colorbar(pclrmsh, cax=axis_colorbar)
    cbar.set_label('-delta T / T')

    # We will modify this line later, and make it visible later
    t0_each_wavelength = np.zeros(len(TA_matrix_wavelengths))     # Values will be entered later
    t0_fit_line, = TA_image_axis_han.plot(TA_matrix_wavelengths, t0_each_wavelength, linewidth=0.7, alpha=0.0)

    # Annotation 
    fig_han.text(0.58, 0.05, 'Instructions:')
    fig_han.text(0.58, 0.03, 'Double Left-Mouse click to move crosshairs.')

    # Plot up the transects
    ax_horiz = fig_han.add_axes([0.06,0.02,0.5,0.23], sharex=TA_image_axis_han)             # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert = fig_han.add_axes([0.625,0.31,0.2,0.68], sharey=TA_image_axis_han)              # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert.axes.yaxis.set_label_text('delay time / ps')

    # Initial Plotting of the crosshairs
    rows, cols = TA_data.shape
    crosshair_v_idx = cols // 2
    crosshair_h_idx = rows-5#*5 // 6
    crosshair_v = TA_image_axis_han.axvline(TA_matrix_wavelengths[crosshair_v_idx], color='k', linestyle='--', alpha=0.2)
    crosshair_h = TA_image_axis_han.axhline(TA_matrix_delay_times[crosshair_h_idx], color='k', linestyle='--', alpha=0.2)


    # Horizontal Transect
    x_indices = np.arange(cols)
    line_horiz, = ax_horiz.plot(TA_matrix_wavelengths, TA_data[crosshair_h_idx,:],  color='k', linewidth=1)
    red_dot_horiz = ax_horiz.scatter(TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx], color='red', s=40, zorder=2)
    ax_horiz.set_ylim([TA_data[crosshair_h_idx,:].min(), TA_data[crosshair_h_idx,:].max()])    
    plt.setp(ax_horiz.get_xticklabels(), visible=False)
    ax_horiz.grid(True, linestyle=':')


    # Vertical Transect
    y_indices = np.arange(rows)
    line_vert, = ax_vert.plot( TA_data[:,crosshair_v_idx], TA_matrix_delay_times, color='k', linewidth=1)
    red_dot_vert = ax_vert.scatter(TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx], color='red', s=40, zorder=2)
    ax_vert.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
    plt.setp(ax_vert.get_yticklabels(), visible=True)
    ax_vert.grid(True, linestyle=':')


    ### Textbox so the user can change the x-axis limits
    wavelength_limits = [np.min(TA_matrix_wavelengths) , np.max(TA_matrix_wavelengths)]
    def w_axis_textbox_update1(expression_str):
        nonlocal wavelength_limits
        wavelength_limits[0] = float(expression_str)
        TA_image_axis_han.axes.set_xlim(wavelength_limits[0], wavelength_limits[1]) 
    text_box_wax1 = fig_han.add_axes([0.02, 0.27, 0.03, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_w1 = TextBox(text_box_wax1, '', initial=str(wavelength_limits[0])[0:5])
    text_box_w1.text_disp.set_fontsize(7), text_box_w1.label.set_position([0,1.5]), text_box_w1.label.set_horizontalalignment('left'), text_box_w1.label.set_fontsize(6)
    text_box_w1.on_submit(w_axis_textbox_update1)
    def w_axis_textbox_update2(expression_str):
        nonlocal wavelength_limits
        wavelength_limits[1] = float(expression_str)
        TA_image_axis_han.axes.set_xlim(wavelength_limits[0], wavelength_limits[1])     
    text_box_wax2 = fig_han.add_axes([0.565, 0.27, 0.03, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_w2 = TextBox(text_box_wax2, '', initial=str(wavelength_limits[1])[0:5])
    text_box_w2.text_disp.set_fontsize(7), text_box_w2.label.set_position([0,1.5]), text_box_w2.label.set_horizontalalignment('left'), text_box_w2.label.set_fontsize(6)
    text_box_w2.on_submit(w_axis_textbox_update2)


    ### Textbox so the user can change the time-axis limit 1
    time_limits = [np.min(TA_matrix_delay_times) , np.max(TA_matrix_delay_times)]
    def t_axis_textbox_update1(expression_str):
        nonlocal time_limits
        time_limits[0] = float(expression_str)
        TA_image_axis_han.axes.set_ylim(time_limits[0], time_limits[1]) 
    text_box_tax1 = fig_han.add_axes([0.004, 0.3, 0.025, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_t1 = TextBox(text_box_tax1, '', initial=str(time_limits[0])[0:4])
    text_box_t1.text_disp.set_fontsize(7), text_box_t1.label.set_position([0,1.7]), text_box_t1.label.set_fontsize(6), text_box_t1.label.set_horizontalalignment('left')
    text_box_t1.on_submit(t_axis_textbox_update1)
    ### Textbox so the user can change the time-axis limit 2
    def t_axis_textbox_update2(expression_str):
        nonlocal time_limits
        time_limits[1] = float(expression_str)
        TA_image_axis_han.axes.set_ylim(time_limits[0], time_limits[1]) 
    text_box_tax2 = fig_han.add_axes([0.004, 0.975, 0.025, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_t2 = TextBox(text_box_tax2, '', initial=str(time_limits[1])[0:4])
    text_box_t2.text_disp.set_fontsize(7)
    text_box_t2.on_submit(t_axis_textbox_update2)


    ### Textbox event handler, so the user can change the colorbar limits
    def colorbar_textbox_update(expression_str):
        min_max = [float(x) for x in expression_str.split(',')]
        TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )
        pclrmsh.set_cmap(TA_colormap)
        pclrmsh.set_clim(min_max)
        # Redraw
        fig_han.canvas.draw_idle()
    text_box_cmap_ax = fig_han.add_axes([0.86, 0.43, 0.13, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_cmap = TextBox(text_box_cmap_ax, 'Manual Colormap Limits', initial=str(min_max[0])[0:9]+' , '+str(min_max[1])[0:9])
    text_box_cmap.text_disp.set_fontsize(7), text_box_cmap.label.set_position([-0.05,1.5]), text_box_cmap.label.set_fontsize(8), text_box_cmap.label.set_horizontalalignment('left')
    text_box_cmap.on_submit(colorbar_textbox_update)


    # Create an interpolator so we can quickly get the value of TA_data at any wavelength or delay_time, and then we print the value of TA_data in the window bar
    TA_matrix_interpolator = RegularGridInterpolator( (TA_matrix_delay_times, np.flip(TA_matrix_wavelengths)), np.fliplr(TA_data), bounds_error=False, fill_value=None)
    TA_image_axis_han.format_coord = lambda x, y: f"crosshairs (w: {TA_matrix_wavelengths[crosshair_v_idx]:0.1f} nm, t: {TA_matrix_delay_times[crosshair_h_idx]:0.1f} ps, z: {TA_data[crosshair_h_idx,crosshair_v_idx]:0.6f})           mouse pointer (w: {x:.1f}, t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f})   "
    ax_horiz.format_coord          = lambda x, y: f"mouse pointer location (w: {x:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"
    ax_vert.format_coord           = lambda x, y: f"mouse pointer location (t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"


    #### MOUSE Click Handler Function
    def on_mouse_click(event):
        nonlocal crosshair_h_idx
        nonlocal crosshair_v_idx
        # Check if click is inside the image axis AND is the Left Button AND is a double-click (We want this to move the crosshairs!)
        if event.inaxes == TA_image_axis_han and event.dblclick and (event.button == 1 or event.button == 3):
            
            # Where was the mouse clicked?
            x_mouse, y_mouse = event.xdata, event.ydata

            # Update Crosshairs to the clicked location
            crosshair_v.set_xdata([x_mouse])
            crosshair_h.set_ydata([y_mouse])

            # Convert to integer indices
            crosshair_v_idx = int(np.abs(TA_matrix_wavelengths - x_mouse).argmin())
            crosshair_h_idx = int(np.abs(TA_matrix_delay_times - y_mouse).argmin())

            # Update the window toolbar message
            if isinstance(event.xdata,float):
                fig_han.canvas.toolbar.set_message( TA_image_axis_han.format_coord(event.xdata, event.ydata) )

            # Obtain the TA_data only in the zoomed view
            xlim = TA_image_axis_han.get_xlim()
            ylim = TA_image_axis_han.get_ylim()
            zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
            zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]

            # Update Horizontal Transect (Bottom Plot)
            # Fetch the column at crosshair_x_idx
            line_horiz.set_ydata(TA_data[crosshair_h_idx, :]) 
            ax_horiz.set_ylim([TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].min(), TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].max()])
            red_dot_horiz.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx]])
            # Reset the averaging to 1
            text_box_horiz_cut_ave.set_val('1')

            # Update Vertical Transect (Side Plot)
            # Fetch the row at crosshair_v_idx
            line_vert.set_xdata(TA_data[:, crosshair_v_idx]) 
            ax_vert.set_xlim([TA_data[min(zoom_t_idx):max(zoom_t_idx),crosshair_v_idx].min(), TA_data[min(zoom_t_idx):max(zoom_t_idx),crosshair_v_idx].max()])    
            red_dot_vert.set_offsets([TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]])
            # Reset the averaging to 1
            text_box_vert_cut_ave.set_val('1')

            # Redraw
            fig_han.canvas.draw_idle()


    #### KEYBOARD BUTTONS event handler functions
    # For our left right keyboard events, we have to disable the normal matplotlib window left-right key bindings 
    print('Disabling normal key bindings for arrows.')
    print("Old back keys:", plt.rcParams['keymap.back'])
    #Remove 'left' and 'right' from the defaults
    # We use a try/pass block just in case you run this code twice (which would cause a ValueError if the key is already gone)
    try:
        plt.rcParams['keymap.back'].remove('left')
        plt.rcParams['keymap.forward'].remove('right')
    except ValueError:
        pass
    mouse_in_plot_axes = False
    def update_mouse_in_plot_axes(event):                # This function keeps track of whether the mouse is inside a textbox
        nonlocal mouse_in_plot_axes
        mouse_in_plot_axes = False
        if event.inaxes == TA_image_axis_han:
            mouse_in_plot_axes = True
    def on_up_down_key_press(event):
        nonlocal crosshair_v_idx
        nonlocal crosshair_h_idx
        if event.key == 'up' and (crosshair_h_idx < len(TA_matrix_delay_times)-1) and mouse_in_plot_axes:
            crosshair_h_idx = crosshair_h_idx + 1
        if event.key == 'down' and crosshair_h_idx > 1 and mouse_in_plot_axes:
            crosshair_h_idx = crosshair_h_idx - 1        
        if event.key == 'left' and (crosshair_v_idx < len(TA_matrix_wavelengths)-1) and mouse_in_plot_axes:
            crosshair_v_idx = crosshair_v_idx + 1
        if event.key == 'right' and crosshair_v_idx > 1 and mouse_in_plot_axes:
            crosshair_v_idx = crosshair_v_idx - 1
        if (event.key == 'up' or event.key == 'down' or event.key == 'left' or event.key == 'right' ) and mouse_in_plot_axes:
            # Update Crosshairs to the clicked location
            crosshair_h.set_ydata([TA_matrix_delay_times[crosshair_h_idx], TA_matrix_delay_times[crosshair_h_idx]])
            crosshair_v.set_xdata([TA_matrix_wavelengths[crosshair_v_idx], TA_matrix_wavelengths[crosshair_v_idx]])
            if isinstance(event.xdata,float):
                fig_han.canvas.toolbar.set_message( TA_image_axis_han.format_coord(event.xdata, event.ydata) )

            # Obtain the TA_data only in the zoomed view
            xlim = TA_image_axis_han.get_xlim()
            ylim = TA_image_axis_han.get_ylim()
            zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
            zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]

            # Update Horizontal Transect (Side Plot)
            # Fetch the column at x_idx
            line_horiz.set_ydata(TA_data[crosshair_h_idx, :]) 
            ax_horiz.set_ylim([TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].min(), TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].max()])    
            red_dot_horiz.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx]])
            # Reset the averaging to 1
            text_box_horiz_cut_ave.set_val('1')

            # Update Vertical Transect (Bottom Plot)
            # Fetch the row at crosshair_h_idx
            line_vert.set_xdata(TA_data[:, crosshair_v_idx])
            ax_vert.set_xlim([TA_data[min(zoom_t_idx):max(zoom_t_idx),crosshair_v_idx].min(), TA_data[min(zoom_t_idx):max(zoom_t_idx),crosshair_v_idx].max()])    
            red_dot_vert.set_offsets([TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]])
            # Reset the averaging to 1
            text_box_vert_cut_ave.set_val('1')

            # Redraw
            fig_han.canvas.draw_idle()      


    #### FIGURE WINDOW BUTTONS Event Handler Functions  
    # A button to update the colormap
    def update_colormap(event):
        # Obtain the TA_data only in the zoomed view
        xlim = TA_image_axis_han.get_xlim()
        ylim = TA_image_axis_han.get_ylim()
        zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
        zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]
        zoomed_data = TA_data[min(zoom_t_idx):max(zoom_t_idx), min(zoom_w_idx):max(zoom_w_idx)]

        # Update the colormap and vmin vmax of the TA_data image
        min_max = [np.percentile(zoomed_data.flatten(),2), np.percentile(zoomed_data.flatten(),98)]
        #min_max = [zoomed_data.min(), zoomed_data.max()]
        TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )
        pclrmsh.set_cmap(TA_colormap)
        pclrmsh.set_clim(min_max)

        #Update the colormap textbox
        text_box_cmap.set_val(str(min_max[0])[0:9]+' , '+str(min_max[1])[0:9])

        # Redraw
        fig_han.canvas.draw_idle()
    ax_button_update = plt.axes([0.87, 0.37, 0.11, 0.03])  # [left, bottom, width, height]
    button_update = Button(ax_button_update, 'Auto Colormap')
    button_update.on_clicked(update_colormap)


    # A button to shift the time axis to linear 
    def linear_time_axis(event):
        TA_image_axis_han.set_yscale('linear')
        fig_han.canvas.draw_idle()
    # Create a button that saves the crosshairs to a history of crosshairs
    ax_linear_time_axis = plt.axes([0.6, 0.197, 0.1, 0.025])  # [left, bottom, width, height]
    button_linear_time_axis = Button(ax_linear_time_axis, 'Linear Axis')
    #button_sym_log_time.text_disp.set_fontsize(7)
    button_linear_time_axis.on_clicked(linear_time_axis) 


    # A button to shift the time axis to symmetric log
    linear_threshold_time_axis = 0.01
    def sym_log_time_axis(event):
        TA_image_axis_han.set_yscale('symlog', linthresh=linear_threshold_time_axis)
        fig_han.canvas.draw_idle()
    # Create a button that saves the crosshairs to a history of crosshairs
    ax_sym_log_time_axis = plt.axes([0.735, 0.197, 0.1, 0.025])  # [left, bottom, width, height]
    button_sym_log_time = Button(ax_sym_log_time_axis, 'Sym Log Axis')
    #button_sym_log_time.text_disp.set_fontsize(7)
    button_sym_log_time.on_clicked(sym_log_time_axis)  
    ### Textbox for the symmetric time axis
    def textbox_sym_log_axis(expression_str):
        nonlocal linear_threshold_time_axis
        linear_threshold_time_axis = float(expression_str)
        TA_image_axis_han.set_yscale('symlog', linthresh=linear_threshold_time_axis)
        fig_han.canvas.draw_idle()
    textbox_sym_log_ax = fig_han.add_axes([0.85, 0.197, 0.05, 0.025])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_sym_log = TextBox(textbox_sym_log_ax, '  Sym Log Axis \nLinear Threshold', initial=str(linear_threshold_time_axis))
    text_box_sym_log.text_disp.set_fontsize(9), text_box_sym_log.label.set_position([-0.25,1.6]), text_box_sym_log.label.set_fontsize(7), text_box_sym_log.label.set_horizontalalignment('left')
    text_box_sym_log.on_submit(textbox_sym_log_axis)


    # Button to average the transects
    linear_threshold_time_axis = 0.01
    def average_transects(event):
        # Obtain the TA_data only in the zoomed view
        xlim = TA_image_axis_han.get_xlim()
        ylim = TA_image_axis_han.get_ylim()
        zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
        zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]
        # Update Horizontal Transect (Bottom Plot)
        # Fetch the column at crosshair_x_idx
        if np.mod((horiz_cut_num_ave-1)/2,2)==0:         # If the crosshairs are too close to the borders of the image then we can't add the extra rows or columns for averaging
            pad = int(np.abs(horiz_cut_num_ave-1)/2)
        else:
            pad = int(np.abs(horiz_cut_num_ave)/2)
        if ((crosshair_h_idx - pad) < 0) or ((crosshair_h_idx + pad +1) >= (len(TA_matrix_delay_times))):
            pad = 0
        averaged_spectrum = np.mean(TA_data[(crosshair_h_idx-pad):(crosshair_h_idx+pad+1), :], axis=0)
        line_horiz.set_ydata(averaged_spectrum) 
        ax_horiz.set_ylim([averaged_spectrum[min(zoom_w_idx):max(zoom_w_idx)].min(), averaged_spectrum[min(zoom_w_idx):max(zoom_w_idx)].max()])
        red_dot_horiz.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], averaged_spectrum[crosshair_v_idx]])
        # Update Vertical Transect (Side Plot)
        # Fetch the row at crosshair_v_idx
        if np.mod((vert_cut_num_ave-1)/2,2)==0:             # If the crosshairs are too close to the borders of the image then we can't add the extra rows or columns for averaging
            pad = int(np.abs(vert_cut_num_ave-1)/2)
        else:
            pad = int(np.abs(vert_cut_num_ave)/2)
        if ((crosshair_v_idx - pad) < 0) or ((crosshair_v_idx + pad +1) >= len(TA_matrix_wavelengths)):
            pad = 0
        averaged_transient = np.mean(TA_data[:, (crosshair_v_idx-pad):(crosshair_v_idx+pad+1)], axis=1)
        line_vert.set_xdata(averaged_transient) 
        ax_vert.set_xlim([averaged_transient[min(zoom_t_idx):max(zoom_t_idx)].min(), averaged_transient[min(zoom_t_idx):max(zoom_t_idx)].max()])    
        red_dot_vert.set_offsets([averaged_transient[crosshair_h_idx], TA_matrix_delay_times[crosshair_h_idx]])
        # Redraw
        fig_han.canvas.draw_idle()
    # Create a button to average the transects
    average_transects_button_ax = plt.axes([0.66, 0.11, 0.135, 0.025])  # [left, bottom, width, height]
    average_transects_button = Button(average_transects_button_ax, 'Average Transects')
    #button_sym_log_time.text_disp.set_fontsize(7)
    average_transects_button.on_clicked(average_transects) 
    # Vertical Transect Textbox so the user can average a chunk of delay_times together in the transect
    fig_han.text(0.598, 0.154, 'num transects to average', fontsize=8)
    vert_cut_num_ave = 1
    def vert_cut_num_ave_update(expression_str):
        nonlocal vert_cut_num_ave
        vert_cut_num_ave = int(expression_str)
    text_box_vert_cut_ave_ax = fig_han.add_axes([0.63, 0.11, 0.025, 0.025])                # [left bottom fraction_vertontal  fraction_vertical ]
    text_box_vert_cut_ave = TextBox(text_box_vert_cut_ave_ax, 'vert', initial='1')
    text_box_vert_cut_ave.text_disp.set_fontsize(9), text_box_vert_cut_ave.label.set_position([-0.1,1.35]), text_box_vert_cut_ave.label.set_fontsize(7), text_box_vert_cut_ave.label.set_horizontalalignment('left')
    text_box_vert_cut_ave.on_submit(vert_cut_num_ave_update)
    # Horizontal Transect Textbox so the user can average a chunk of delay_times together in the transect
    horiz_cut_num_ave = 1
    def horiz_cut_num_ave_update(expression_str):
        nonlocal horiz_cut_num_ave
        horiz_cut_num_ave = int(expression_str)
    text_box_horiz_cut_ave_ax = fig_han.add_axes([0.6, 0.11, 0.025, 0.025])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_horiz_cut_ave = TextBox(text_box_horiz_cut_ave_ax, 'horz', initial='1')
    text_box_horiz_cut_ave.text_disp.set_fontsize(9), text_box_horiz_cut_ave.label.set_position([0,1.35]), text_box_horiz_cut_ave.label.set_fontsize(7), text_box_horiz_cut_ave.label.set_horizontalalignment('left')
    text_box_horiz_cut_ave.on_submit(horiz_cut_num_ave_update)


    # A button to save the plotted time transient decay
    def save_TA_transient(event):
        if isinstance(TA_matrix_input, str):
            output_filename = TA_matrix_input + '.transient' + str(TA_matrix_wavelengths[crosshair_v_idx])[0:6] + 'nm.csv'
        else:
            output_filename = 'transient' + str(TA_matrix_wavelengths[crosshair_v_idx])[0:6] + 'nm.csv'
        #Save the data
        np.savetxt(output_filename, np.hstack((line_vert.get_xdata(),line_vert.get_ydata())), delimiter=',')
    # Create a button that saves the crosshairs to a history of crosshairs
    ax_save_transient = plt.axes([0.82, 0.11, 0.16, 0.025])  # [left, bottom, width, height]
    button_save_transient = Button(ax_save_transient, 'Save Plotted Transient')
    #button_sym_log_time.text_disp.set_fontsize(7)
    button_save_transient.on_clicked(save_TA_transient)  


    # Event handler for mouse clicking
    fig_han.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Event handler for up down buttons
    fig_han.canvas.mpl_connect('key_press_event', on_up_down_key_press)
 
    # Event handler to check if the mouse is in the plot window
    fig_han.canvas.mpl_connect('motion_notify_event', update_mouse_in_plot_axes)

    plt.show(block=True)        # this is needed to stop the program from executing the return immediately









