
#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import h5py
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.widgets import RadioButtons
from IPython.display import display
import time

# Import our custom codes
import shared_supporting_functions as TA_sh




# This function requires TA_matrix to hold a row vector of wavelengths in row 1, and a vector of probe delay times in column 1.
def merge_TA_matrix_GUI(list_of_TA_matrix_filenames):
    print('NOTE: This function always merges TA data into the format of the first TA_matrix in the input list.')

    # Get the TA_matrix data
    TA_matrix = TA_sh.get_TA_matrix(list_of_TA_matrix_filenames[0])
    output_filename = list_of_TA_matrix_filenames[0]+'.merged.csv'


    # Extract the wavelengths and time delays
    TA_matrix_wavelengths = TA_matrix[0,1:]
    TA_matrix_delay_times = TA_matrix[1:,0]
    # Crop the TA image down to remove the wavelengths and delay times
    TA_data = TA_matrix[1:,1:]


    # Create a good color map
    min_max = [np.percentile(TA_data.flatten(),2), np.percentile(TA_data.flatten(),98)]
    TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )

    # Create figure
    fig_han = plt.figure(figsize=(12, 9.7))                        #fig_han.get_size_inches()
    fig_han.tight_layout()

    # Plot up the main TA matrix image and colorbar
    TA_image_axis_han = fig_han.add_axes([0.07,0.435,0.5,0.561])            # [left bottom fraction_horizontal  fraction_vertical ]
    pclrmsh = TA_image_axis_han.pcolormesh(TA_matrix_wavelengths, TA_matrix_delay_times, TA_data, vmin=min_max[0], vmax=min_max[1], cmap=TA_colormap, mouseover=True)
    TA_image_axis_han.axes.yaxis.set_label_text('delay time / ps')
    TA_image_axis_han.axes.xaxis.set_label_text('wavelength / nm')
    axis_colorbar = fig_han.add_axes([0.87, 0.58, 0.02, 0.412])
    cbar = fig_han.colorbar(pclrmsh, cax=axis_colorbar)
    cbar.set_label('-delta T / T')

    # We will modify this line later, and make it visible later
    t0_each_wavelength = np.zeros(len(TA_matrix_wavelengths))     # Values will be entered later
    t0_fit_line, = TA_image_axis_han.plot(TA_matrix_wavelengths, t0_each_wavelength, linewidth=0.7, alpha=0.0)

    # Annotation 
    fig_han.text(0.68, 0.065, 'Instructions:')
    fig_han.text(0.68, 0.045, 'Double Left-Mouse click to move crosshairs.')
    fig_han.text(0.68, 0.025, 'Keyboard arrow buttons also move crosshairs.')

    # Axes for the transects
    ax_horiz = fig_han.add_axes([0.07,0.19,0.5,0.1896], sharex=TA_image_axis_han)             # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert = fig_han.add_axes([0.625,0.435,0.2,0.56], sharey=TA_image_axis_han)              # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert.axes.yaxis.set_label_text('delay time / ps')

    # Plot of the crosshairs
    rows, cols = TA_data.shape
    crosshair_v_idx = cols // 2
    crosshair_h_idx = rows-5#*5 // 6
    crosshair_v_wavelength = TA_matrix_wavelengths[crosshair_v_idx]
    crosshair_h_delay_time = TA_matrix_delay_times[crosshair_h_idx]
    crosshair_v = TA_image_axis_han.axvline(crosshair_v_wavelength, color='k', linestyle='--', alpha=0.2)
    crosshair_h = TA_image_axis_han.axhline(TA_matrix_delay_times[crosshair_h_idx], color='k', linestyle='--', alpha=0.2)

    # Plot of Horizontal Transect
    line_horiz, = ax_horiz.plot(TA_matrix_wavelengths, TA_data[crosshair_h_idx,:],  color='k', linewidth=1)
    red_dot_horiz = ax_horiz.scatter(TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx], color='red', s=40, zorder=2)
    ax_horiz.set_ylim([TA_data[crosshair_h_idx,:].min(), TA_data[crosshair_h_idx,:].max()])    
    plt.setp(ax_horiz.get_xticklabels(), visible=False)
    ax_horiz.grid(True, linestyle=':')

    # Plot of Initial Vertical Transect
    line_vert, = ax_vert.plot( TA_data[:,crosshair_v_idx], TA_matrix_delay_times, color='k', linewidth=1)
    red_dot_vert = ax_vert.scatter(TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx], color='red', s=40, zorder=2)
    ax_vert.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
    plt.setp(ax_vert.get_yticklabels(), visible=True)
    ax_vert.grid(True, linestyle=':')


    # Calculate the fractions of each spectrum in the final merged spectra
    all_probe_counts = np.zeros( (len(list_of_TA_matrix_filenames), len(TA_matrix_wavelengths) ) )
    # Interpolate each probe count to
    print('WARNING: The minimum probe count will be subtracted from each probe spectrum to clear background noise.')
    for i in range(0, len(list_of_TA_matrix_filenames)):
        raw_wavelength =   np.flip( TA_sh.get_TA_matrix( list_of_TA_matrix_filenames[i])[0,1:] )
        raw_probe_counts = np.flip( TA_sh.get_TA_probe_counts(list_of_TA_matrix_filenames[i]) )
        raw_probe_count_interpolator = CubicSpline(raw_wavelength, raw_probe_counts, extrapolate = True)
        all_probe_counts[i,:] = raw_probe_count_interpolator(TA_matrix_wavelengths)
        all_probe_counts[i,:] = all_probe_counts[i,:] - np.min(all_probe_counts[i,:])
    sqrt_all_probe_counts = np.sqrt(all_probe_counts)
    fraction_in_merge_each_probe_spectrum = sqrt_all_probe_counts.copy()
    for i in range(0, len(list_of_TA_matrix_filenames)):
        fraction_in_merge_each_probe_spectrum[i,:] =  sqrt_all_probe_counts[i,:] / np.sum(sqrt_all_probe_counts, axis=0)


    # Plot the probe counts and fraction in merge
    probe_counts = TA_sh.get_TA_probe_counts(list_of_TA_matrix_filenames[0])
    ax_probe_counts = fig_han.add_axes([0.07,0.01,0.5,0.161], sharex=TA_image_axis_han)              # [left bottom fraction_horizontal  fraction_vertical ]
    ax_probe_counts.axes.yaxis.set_label_text('probe counts', color='b')
    ax_probe_counts.tick_params(axis='y', colors='blue')
    line_probe_counts, = ax_probe_counts.plot(TA_matrix_wavelengths, probe_counts, color='b')
    ax_probe_counts.set_ylim(0,np.max(probe_counts))
    red_dot_probe_counts = ax_probe_counts.scatter(TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx], color='red', s=40, zorder=2)
    ax_fraction_in_merge = ax_probe_counts.twinx()
    line_fraction_in_merge, = ax_fraction_in_merge.plot(TA_matrix_wavelengths, fraction_in_merge_each_probe_spectrum[0,:], color='k')
    ax_fraction_in_merge.axes.yaxis.set_label_text('fraction in merge')
    ax_fraction_in_merge.set_ylim(0,1)
    plt.setp(ax_probe_counts.get_xticklabels(), visible=False)
    ax_probe_counts.grid(True, linestyle=':')


    # Create an interpolator so we can quickly get the value of TA_data at any wavelength or delay_time, and then we print the value of TA_data in the window bar
    TA_matrix_interpolator = RegularGridInterpolator( (TA_matrix_delay_times, np.flip(TA_matrix_wavelengths)), np.fliplr(TA_data), bounds_error=False, fill_value=None)
    TA_image_axis_han.format_coord = lambda x, y: f"crosshairs (w: {TA_matrix_wavelengths[crosshair_v_idx]:0.1f} nm, t: {TA_matrix_delay_times[crosshair_h_idx]:0.1f} ps, z: {TA_data[crosshair_h_idx,crosshair_v_idx]:0.6f})           mouse pointer (w: {x:.1f}, t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f})   "
    ax_horiz.format_coord          = lambda x, y: f"mouse pointer location (w: {x:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"
    ax_vert.format_coord           = lambda x, y: f"mouse pointer location (t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"



    ### Textbox so the user can change the x-axis limits
    wavelength_limits = [np.min(TA_matrix_wavelengths) , np.max(TA_matrix_wavelengths)]
    def w_axis_textbox_update1(expression_str):
        nonlocal wavelength_limits
        wavelength_limits[0] = float(expression_str)
        TA_image_axis_han.axes.set_xlim(wavelength_limits[0], wavelength_limits[1]) 
    text_box_wax1 = fig_han.add_axes([0.04, 0.4, 0.03, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_w1 = TextBox(text_box_wax1, '', initial=str(wavelength_limits[0])[0:5])
    text_box_w1.text_disp.set_fontsize(7), text_box_w1.label.set_position([0,1.5]), text_box_w1.label.set_horizontalalignment('left'), text_box_w1.label.set_fontsize(6)
    text_box_w1.on_submit(w_axis_textbox_update1)
    def w_axis_textbox_update2(expression_str):
        nonlocal wavelength_limits
        wavelength_limits[1] = float(expression_str)
        TA_image_axis_han.axes.set_xlim(wavelength_limits[0], wavelength_limits[1])     
    text_box_wax2 = fig_han.add_axes([0.575, 0.4, 0.03, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_w2 = TextBox(text_box_wax2, '', initial=str(wavelength_limits[1])[0:5])
    text_box_w2.text_disp.set_fontsize(7), text_box_w2.label.set_position([0,1.5]), text_box_w2.label.set_horizontalalignment('left'), text_box_w2.label.set_fontsize(6)
    text_box_w2.on_submit(w_axis_textbox_update2)


    ### Textbox so the user can change the time-axis limit 1
    time_limits = [np.min(TA_matrix_delay_times) , np.max(TA_matrix_delay_times)]
    def t_axis_textbox_update1(expression_str):
        nonlocal time_limits
        time_limits[0] = float(expression_str)
        TA_image_axis_han.axes.set_ylim(time_limits[0], time_limits[1]) 
    text_box_tax1 = fig_han.add_axes([0.004, 0.43, 0.025, 0.02])                # [left bottom fraction_horizontal  fraction_vertical ]
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
        try:
            min_max = np.array(expression_str.split(','))
        except:
            Print('You entered the data in the textbox incorrectly. It should be two numbers separated by a comma.')
            pass
        else:
            min_max = [float(x) for x in expression_str.split(',')]
            TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )
            pclrmsh.set_cmap(TA_colormap)
            pclrmsh.set_clim(min_max)
            # Redraw
            fig_han.canvas.draw_idle()
    text_box_cmap_ax = fig_han.add_axes([0.86, 0.54, 0.13, 0.0165])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_cmap = TextBox(text_box_cmap_ax, 'Manual Colormap Limits', initial=str(min_max[0])[0:7]+' , '+str(min_max[1])[0:7])
    text_box_cmap.text_disp.set_fontsize(7), text_box_cmap.label.set_position([0,1.5]), text_box_cmap.label.set_fontsize(8), text_box_cmap.label.set_horizontalalignment('left')
    text_box_cmap.on_submit(colorbar_textbox_update)


    #### MOUSE Click Handler Function
    def on_mouse_click(event):
        nonlocal crosshair_h_idx
        nonlocal crosshair_v_idx
        nonlocal crosshair_v_wavelength
        nonlocal crosshair_h_delay_time
        # Check if click is inside the image axis AND is the Left Button AND is a double-click (We want this to move the crosshairs!)
        if event.inaxes == TA_image_axis_han and event.dblclick and (event.button == 1 or event.button == 3):
            
            # Where was the mouse clicked?
            x_mouse, y_mouse = event.xdata, event.ydata
            crosshair_v_wavelength = x_mouse
            crosshair_h_delay_time = y_mouse

            # Update Crosshairs to the clicked location
            crosshair_v.set_xdata([crosshair_v_wavelength])
            crosshair_h.set_ydata([crosshair_h_delay_time])

            # Convert to integer indices
            crosshair_v_idx = int(np.abs(TA_matrix_wavelengths - crosshair_v_wavelength).argmin())
            crosshair_h_idx = int(np.abs(TA_matrix_delay_times - crosshair_h_delay_time).argmin())

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

            # Update Vertical Transect (Side Plot)
            # Fetch the row at crosshair_h_idx
            line_vert.set_xdata(TA_data[:, crosshair_v_idx])
            ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
            red_dot_vert.set_offsets([TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]])

            # Draw the red dot on the probe counts
            red_dot_probe_counts.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx]])

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
        nonlocal crosshair_v_wavelength
        nonlocal crosshair_h_delay_time
        if event.key == 'up' and (crosshair_h_idx < len(TA_matrix_delay_times)-1) and mouse_in_plot_axes:
            crosshair_h_idx = crosshair_h_idx + 1
        if event.key == 'down' and crosshair_h_idx > 1 and mouse_in_plot_axes:
            crosshair_h_idx = crosshair_h_idx - 1        
        if event.key == 'left' and (crosshair_v_idx < len(TA_matrix_wavelengths)-1) and mouse_in_plot_axes:
            crosshair_v_idx = crosshair_v_idx + 1
        if event.key == 'right' and crosshair_v_idx > 1 and mouse_in_plot_axes:
            crosshair_v_idx = crosshair_v_idx - 1
        if (event.key == 'up' or event.key == 'down' or event.key == 'left' or event.key == 'right' ):
            # Update Crosshairs to the clicked location
            crosshair_v_wavelength = TA_matrix_wavelengths[crosshair_v_idx]
            crosshair_h_delay_time = TA_matrix_delay_times[crosshair_h_idx]
            crosshair_h.set_ydata([crosshair_h_delay_time , crosshair_h_delay_time])
            crosshair_v.set_xdata([crosshair_v_wavelength , crosshair_v_wavelength])
            # Update the window toolbar message
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

            # Update Vertical Transect (Bottom Plot)
            # Fetch the row at crosshair_h_idx
            line_vert.set_xdata(TA_data[:, crosshair_v_idx])
            ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
            red_dot_vert.set_offsets([TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]])

            # Update the Red Dot on the probe counts plot
            red_dot_probe_counts.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx]])

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
    ax_button_update = plt.axes([0.87, 0.498, 0.11, 0.0247])  # [left, bottom, width, height]
    button_update = Button(ax_button_update, 'Auto Colormap')
    button_update.on_clicked(update_colormap)



    # A radio List to select a different TA_matrix
    radio_options = list_of_TA_matrix_filenames
    presently_RadioClicked_TA_dataset = 0
    def update_plot_via_radio_list(label):
        nonlocal TA_matrix
        nonlocal TA_matrix_wavelengths
        nonlocal TA_matrix_delay_times
        nonlocal TA_data
        nonlocal pclrmsh
        nonlocal min_max
        nonlocal TA_colormap
        nonlocal TA_matrix_interpolator
        nonlocal crosshair_v_idx
        nonlocal crosshair_h_idx
        nonlocal line_horiz
        nonlocal line_vert
        nonlocal red_dot_horiz
        nonlocal red_dot_vert
        nonlocal probe_counts 
        nonlocal presently_RadioClicked_TA_dataset
        presently_RadioClicked_TA_dataset = radio_options.index(label)
        TA_matrix = TA_sh.get_TA_matrix(label)
        # Extract the wavelengths and time delays
        TA_matrix_wavelengths = TA_matrix[0,1:]
        TA_matrix_delay_times = TA_matrix[1:,0]
        # Crop the TA image down to remove the wavelengths and delay times
        TA_data = TA_matrix[1:,1:]
        # Create a good color map
        min_max = [np.percentile(TA_data.flatten(),2), np.percentile(TA_data.flatten(),98)]
        TA_colormap = TA_sh.create_TA_Blue_White_Red_Black_colormap( min_max )
        #pclrmsh.set_array(TA_data.ravel())
        pclrmsh.remove()
        pclrmsh = TA_image_axis_han.pcolormesh(TA_matrix_wavelengths, TA_matrix_delay_times, TA_data, vmin=min_max[0], vmax=min_max[1], cmap=TA_colormap, mouseover=True)
        # Get the indices of the crosshairs in the new TA_data matrix
        crosshair_v_idx = int(np.abs(TA_matrix_wavelengths - crosshair_v_wavelength).argmin())
        crosshair_h_idx = int(np.abs(TA_matrix_delay_times - crosshair_h_delay_time).argmin())
        # Create an interpolator so we can quickly get the value of TA_data at any wavelength or delay_time, and then we print the value of TA_data in the window bar
        TA_matrix_interpolator = RegularGridInterpolator( (TA_matrix_delay_times, np.flip(TA_matrix_wavelengths)), np.fliplr(TA_data), bounds_error=False, fill_value=None)
        TA_image_axis_han.format_coord = lambda x, y: f"crosshairs (w: {TA_matrix_wavelengths[crosshair_v_idx]:0.1f} nm, t: {TA_matrix_delay_times[crosshair_h_idx]:0.1f} ps, z: {TA_data[crosshair_h_idx,crosshair_v_idx]:0.6f})           mouse pointer (w: {x:.1f}, t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f})   "
        ax_horiz.format_coord          = lambda x, y: f"mouse pointer location (w: {x:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"
        ax_vert.format_coord           = lambda x, y: f"mouse pointer location (t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"
        # Obtain the TA_data only in the zoomed view
        xlim = TA_image_axis_han.get_xlim()
        ylim = TA_image_axis_han.get_ylim()
        zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
        zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]
        # Update Horizontal Transect (Bottom Plot)
        line_horiz.remove()
        line_horiz, = ax_horiz.plot(TA_matrix_wavelengths, TA_data[crosshair_h_idx,:],  color='k', linewidth=1)
        red_dot_horiz.remove()
        red_dot_horiz = ax_horiz.scatter(TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx], color='red', s=40, zorder=2)
        # Update Vertical Transect (Side Plot)
        line_vert.remove()
        line_vert, = ax_vert.plot( TA_data[:,crosshair_v_idx], TA_matrix_delay_times, color='k', linewidth=1)
        red_dot_vert.remove()
        red_dot_vert = ax_vert.scatter(TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx], color='red', s=40, zorder=2)
        # Draw the red dot on the probe counts
        probe_counts = all_probe_counts[presently_RadioClicked_TA_dataset,:]
        line_probe_counts.set_ydata(probe_counts) 
        ax_probe_counts.set_ylim(0,np.max(probe_counts))
        red_dot_probe_counts.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx]])
        line_fraction_in_merge.set_ydata(fraction_in_merge_each_probe_spectrum[presently_RadioClicked_TA_dataset ,:])
        fig_han.canvas.draw_idle()          # Redraw the figure
    ax_radio = plt.axes([0.57, 0.33, 0.3, 0.02*len(list_of_TA_matrix_filenames)])      # [left, bottom, width, height]
    ax_radio.set_frame_on(False)
    radio_list = RadioButtons(ax_radio, radio_options)
    for label in radio_list.labels:
        label.set_fontsize(8) 
    radio_list.on_clicked(update_plot_via_radio_list)



    # A button to Zero Out a Portion of the Probe Counts Spectrum
    probe_counts_zero_w_range='0,0'
    def button_zero_probe_counts_execute(event):
        nonlocal all_probe_counts
        nonlocal probe_counts
        nonlocal fraction_in_merge_each_probe_spectrum
        nonlocal sqrt_all_probe_counts
        nonlocal probe_counts_zero_w_range
        w_range = np.array(probe_counts_zero_w_range.split(',')).astype(float)
        if (np.sum(np.isnan(w_range)) == 0 and np.sum(probe_counts_zero_w_range == 0).any() == False and w_range[1]>w_range[0]):
            w_idxs = [ np.abs(TA_matrix_wavelengths - w_range[0]).argmin(), np.abs(TA_matrix_wavelengths - w_range[1]).argmin() ]
            all_probe_counts[presently_RadioClicked_TA_dataset, np.min(w_idxs):np.max(w_idxs)] = 0.0000000001
            sqrt_all_probe_counts[:] = np.sqrt(all_probe_counts)
            for i in range(0, len(list_of_TA_matrix_filenames)):
                fraction_in_merge_each_probe_spectrum[i,:] =  sqrt_all_probe_counts[i,:] / np.sum(sqrt_all_probe_counts, axis=0)
            probe_counts[:] = all_probe_counts[presently_RadioClicked_TA_dataset,:]
            line_probe_counts.set_ydata(probe_counts) 
            red_dot_probe_counts.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx]])
            line_fraction_in_merge.set_ydata(fraction_in_merge_each_probe_spectrum[presently_RadioClicked_TA_dataset , :])
            fig_han.canvas.draw_idle()
    ax_button_zero_probe_counts = plt.axes([0.63, 0.25, 0.16, 0.023])  # [left, bottom, width, height]
    button_zero_probe_counts = Button(ax_button_zero_probe_counts, 'Zero Out Counts in Range')
    button_zero_probe_counts.label.set_fontsize(9)
    button_zero_probe_counts.on_clicked(button_zero_probe_counts_execute) 

    ### Textbox for the Zero Out a Portion of the Probe Counts Spectrum
    def execute_zero_probe_counts_range(expression_str):
        nonlocal probe_counts_zero_w_range
        try:
            w_range = np.array(probe_counts_zero_w_range.split(',')).astype(float)
        except:
            Print('You entered the data in the textbox incorrectly. It should be two numbers separated by a comma.')
        else:
            probe_counts_zero_w_range = expression_str
    textbox_zero_probe_counts_range_ax = fig_han.add_axes([0.81, 0.25, 0.08, 0.023])                # [left bottom fraction_horizontal  fraction_vertical ]
    textbox_zero_probe_counts_range = TextBox(textbox_zero_probe_counts_range_ax, 'Range of Wavelengths to Zero', initial=probe_counts_zero_w_range)
    textbox_zero_probe_counts_range.text_disp.set_fontsize(9), textbox_zero_probe_counts_range.label.set_position([-0.05,1.5]), textbox_zero_probe_counts_range.label.set_fontsize(8), textbox_zero_probe_counts_range.label.set_horizontalalignment('left')
    textbox_zero_probe_counts_range.on_submit(execute_zero_probe_counts_range)



   # A button to Smooth the Fraction to Merge of Probe Counts Spectrum
    def smooth_the_merge(event):
        nonlocal all_probe_counts
        nonlocal probe_counts
        nonlocal fraction_in_merge_each_probe_spectrum
        nonlocal sqrt_all_probe_counts
        nonlocal probe_counts_zero_w_range
        w_range = np.array(probe_counts_zero_w_range.split(',')).astype(float)
        if (np.sum(np.isnan(w_range)) == 0 and np.sum(probe_counts_zero_w_range == 0).any() == False and w_range[1]>w_range[0]):      #Make sure the user has entered valid wavelengths to zero out.
            w_idxs = [ np.abs(TA_matrix_wavelengths - w_range[0]).argmin(), np.abs(TA_matrix_wavelengths - w_range[1]).argmin() ]
            for w_idx in w_idxs:
                # Smooth the probe counts
                smoothing_window_size = int( 2 / ((np.max(TA_matrix_wavelengths) - np.min(TA_matrix_wavelengths))/len(TA_matrix_wavelengths)) )  #This should make the window size 2 nm approximately
                if (w_idx>(2*smoothing_window_size)) and (w_idx<len(TA_matrix_wavelengths)-2*smoothing_window_size) :                                                     # We don't want to attempt smoothing at locations close to the edges of the wavelength range
                    for ww_idx in range(w_idx - smoothing_window_size, w_idx + smoothing_window_size):                                                                                   # Repeat 4 times to smooth more
                        all_probe_counts[presently_RadioClicked_TA_dataset, ww_idx] = np.sum(all_probe_counts[presently_RadioClicked_TA_dataset, (ww_idx - smoothing_window_size):(ww_idx + smoothing_window_size)]) / (2*smoothing_window_size)
                    sqrt_all_probe_counts[:] = np.sqrt(all_probe_counts)
                    for i in range(0, len(list_of_TA_matrix_filenames)):
                        fraction_in_merge_each_probe_spectrum[i,:] =  sqrt_all_probe_counts[i,:] / np.sum(sqrt_all_probe_counts, axis=0)
                    probe_counts[:] = all_probe_counts[presently_RadioClicked_TA_dataset,:]
                    line_probe_counts.set_ydata(probe_counts) 
                    red_dot_probe_counts.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], probe_counts[crosshair_v_idx]])
                    line_fraction_in_merge.set_ydata(fraction_in_merge_each_probe_spectrum[presently_RadioClicked_TA_dataset , :])
                    fig_han.canvas.draw_idle() 
    button_smooth_the_merge_ax = plt.axes([0.63, 0.21, 0.16, 0.023])  # [left, bottom, width, height]
    button_smooth_the_merge = Button(button_smooth_the_merge_ax, 'Smooth the Merge Fraction')
    button_smooth_the_merge.label.set_fontsize(9)
    button_smooth_the_merge.on_clicked(smooth_the_merge) 



    # A button to Merge the TA data!
    first_TA_matrix = TA_sh.get_TA_matrix(list_of_TA_matrix_filenames[0])
    merged_TA_data = first_TA_matrix.copy() 
    def merge_TA_data(event):
        nonlocal merged_TA_data
        final_merged_wavelengths = first_TA_matrix[0,1:]
        final_merged_delay_times = first_TA_matrix[1:,0]
        merged_TA_data[1:,1:] = first_TA_matrix[1:,1:] * fraction_in_merge_each_probe_spectrum[0,:]
        for i in range(1, len(list_of_TA_matrix_filenames)):
            next_TA_matrix = TA_sh.get_TA_matrix(list_of_TA_matrix_filenames[i])
            next_merged_wavelengths = next_TA_matrix[0,1:]
            next_merged_delay_times = next_TA_matrix[1:,0]
            if (np.abs(next_merged_wavelengths - final_merged_wavelengths)>0.01).any() or (np.abs(next_merged_delay_times - final_merged_delay_times)>0.1).any():
                merged_TA_data[1:,1:] = merged_TA_data[1:,1:] +  Interpolate_TA_matrix(list_of_TA_matrix_filenames[i], final_merged_wavelengths, final_merged_delay_times)[1:,1:] * fraction_in_merge_each_probe_spectrum[i,:]
            else:
                print('Interpolation not necessary.')
                merged_TA_data[1:,1:] = merged_TA_data[1:,1:] + next_TA_matrix[1:,1:] * fraction_in_merge_each_probe_spectrum[i,:]
            save_merged_TA_matrix_to_disk()
    ax_merge_TA_data = plt.axes([0.81, 0.21, 0.13, 0.023])  # [left, bottom, width, height]
    button_merge_TA_data = Button(ax_merge_TA_data, 'Merge TA datasets')
    button_merge_TA_data.on_clicked(merge_TA_data) 



    def save_merged_TA_matrix_to_disk():
        np.savetxt(output_filename, merged_TA_data, delimiter=',')
    # Create a button to saves the TA_matrix to disk
    #ax_button_save_TA_to_disk = plt.axes([0.81, 0.13, 0.12, 0.023])  # [left, bottom, width, height]
    #Button_save_TA_to_disk = Button(ax_button_save_TA_to_disk, 'Save TA to disk')
    #Button_save_TA_to_disk.on_clicked(save_merged_TA_matrix_to_disk) 


    # Event handler for mouse clicking
    fig_han.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Event handler for up down buttons
    fig_han.canvas.mpl_connect('key_press_event', on_up_down_key_press)

    # Event handler to check if the mouse is in the plot window
    fig_han.canvas.mpl_connect('motion_notify_event', update_mouse_in_plot_axes)
 
    plt.show(block=True)        # this is needed to stop the program from executing the return immediately







def merge_second_TA_matrix_into_first_TA_matrix(list_of_TA_matrix_filenames):
    first_TA_matrix = TA_sh.get_TA_matrix(list_of_TA_matrix_filenames[0])
    final_wavelengths = first_TA_matrix[0,1:]
    final_delay_times = first_TA_matrix[1:,0]

    merged_second_TA_matrix = Interpolate_TA_matrix(list_of_TA_matrix_filenames[0], final_wavelengths, final_delay_times)

    return merged_second_TA_matrix












def merge_TA_matrices(list_of_TA_matrix_filenames, final_wavelengths, final_delay_times):

    # Read the Igor Processed CSV spectra
    Sfeir_blue_TA_image = np.loadtxt(Sfeir_TA_blue_spectrum_IgorProcessed_filename, delimiter=',',  ndmin=2)
    Sfeir_red_TA_image = np.loadtxt(Sfeir_TA_red_spectrum_IgorProcessed_filename, delimiter=',', ndmin=2)

    # Interpolate the blue spectra onto the (time, wavelength) points of the red spectra
    Interpolated_blue_TA_data = Interpolate_TA_matrix(Sfeir_blue_TA_image, final_wavelengths, final_delay_times)
    Interpolated_red_TA_data = Interpolate_TA_matrix(Sfeir_red_TA_image, final_wavelengths, final_delay_times)

    merged_Sfeir_TA = np.zeros([len(final_delay_times)+1, len(final_wavelengths)+1])
    for i in range(0,len(final_delay_times)): merged_Sfeir_TA[i+1,0] = final_delay_times[i]
    for j in range(0,len(final_wavelengths)): merged_Sfeir_TA[0,j+1] = final_wavelengths[j]
    merged_Sfeir_TA[1:,1:] = merge_percentage_red * Interpolated_red_TA_data[1:,1:].T + (1 - merge_percentage_red) * Interpolated_blue_TA_data[1:,1:].T

    return merged_Sfeir_TA








def Interpolate_TA_matrix(TA_matrix_input, new_ws, new_ts):
    TA_matrix = TA_sh.get_TA_matrix(TA_matrix_input)
    print('interpolate subfunc')  

    old_ws = TA_matrix[0,1:]
    old_ts = TA_matrix[1:,0]
    
    # Create the Interpolator
    # bounds_error=False allows extrapolation or handling points outside the grid without crashing
    # fill_value=None tells it to extrapolate values outside the bounds
    print('Starting an Interpolation. This will take a minute or two. Be patient.')
    TA_matrix_interpolator = RegularGridInterpolator((old_ts, old_ws), TA_matrix[1:,1:], bounds_error=False, fill_value=None)

    TA_matrix_interpolated = np.zeros([len(new_ts)+1, len(new_ws)+1])

    # Enter the wavelengths into the first row
    for j in range(0 , len(new_ws)):
        TA_matrix_interpolated[0,j+1] = new_ws[j]

    # Enter the delay times into the first row
    for i in range(0 , len(new_ts)):
        TA_matrix_interpolated[i+1,0] = new_ts[i]

    # Enter the interpolated values
    for i in range(0 , len(new_ts)):
        for j in range(0 , len(new_ws)):
            TA_matrix_interpolated[i+1,j+1] = TA_matrix_interpolator( [ new_ts[i], new_ws[j] ] )[0]

    # output / Save Results
    print("Interpolation successful.")
    
    # Optional: Save results to a new CSV
    #np.savetxt('out.csv', target_TA_matrix, delimiter=',')
    #print("Results saved to 'results.csv'")
    return TA_matrix_interpolated






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




