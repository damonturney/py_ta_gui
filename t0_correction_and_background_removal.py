#!/usr/bin/env python                    #this enables a user to run the file by typing only it's name (no need for python prefix)

"""    block comment
Created 20251202

@author: dturney
"""


# TA_data are assumed to be 2D matrices with wavelengths along the 1st row, and with probe delay times along the 1st column

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

# Import our custom codes
import shared_supporting_functions as TA_sh

plt.close('all')

# This function requires TA_matrix to hold a row vector of wavelengths in row 1, and a vector of probe delay times in column 1.
def t0_correction_and_background_removal(TA_matrix_input):

    # Output filename
    if  isinstance(TA_matrix_input,str):
        output_filename = TA_matrix_input + '.t0_corr.csv'
    else:
        output_filename = 'output.t0_corr.csv'

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
    t0_fit_line, = TA_image_axis_han.plot(TA_matrix_wavelengths, t0_each_wavelength, linewidth=0.7, alpha=0.0, color='k')

    # Annotation 
    fig_han.text(0.58, 0.065, 'Instructions:')
    fig_han.text(0.58, 0.045, 'Double Left-Mouse click to move crosshairs.')
    fig_han.text(0.58, 0.025, 'Double Right-Mouse (or Button) click to Save Crosshairs for t0 fit.')

    # Plot up the transects
    ax_horiz = fig_han.add_axes([0.06,0.02,0.5,0.23], sharex=TA_image_axis_han)             # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert = fig_han.add_axes([0.625,0.31,0.2,0.68], sharey=TA_image_axis_han)              # [left bottom fraction_horizontal  fraction_vertical ]
    ax_vert.axes.yaxis.set_label_text('delay time / ps')

    # 2. Initial Plotting of the crosshairs
    rows, cols = TA_data.shape
    crosshair_h_idx = rows*5 // 6
    crosshair_v_idx = cols // 3
    crosshair_v = TA_image_axis_han.axvline(TA_matrix_wavelengths[crosshair_v_idx], color='k', linestyle='--', alpha=0.2)
    crosshair_h = TA_image_axis_han.axhline(TA_matrix_delay_times[crosshair_h_idx], color='k', linestyle='--', alpha=0.2)

    # Initial Horizontal Transect
    x_indices = np.arange(cols)
    line_horiz, = ax_horiz.plot(TA_matrix_wavelengths, TA_data[crosshair_h_idx,:],  color='k', linewidth=1)
    red_dot_horiz = ax_horiz.scatter(TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx], color='red', s=40, zorder=2)
    ax_horiz.set_ylim([TA_data[crosshair_h_idx,:].min(), TA_data[crosshair_h_idx,:].max()])    
    plt.setp(ax_horiz.get_xticklabels(), visible=False)
    ax_horiz.grid(True, linestyle=':')

    # Initial Vertical Transect
    y_indices = np.arange(rows)
    line_vert, = ax_vert.plot( TA_data[:,crosshair_v_idx], TA_matrix_delay_times, color='k', linewidth=1)
    red_dot_vert = ax_vert.scatter(TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx], color='red', s=40, zorder=2)
    ax_vert.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
    plt.setp(ax_vert.get_yticklabels(), visible=True)
    ax_vert.grid(True, linestyle=':')

    # Create a list to hold a history of clicked points
    click_history = [[-10000,-10000]]    # start with a value of zero in there to allow yourself to check if values are increasing, then cut away this zero for final fitting

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


    ### Textbox event handler, so the user can name the output file
    def output_filename_textbox_update(expression_str):
        output_filename = expression_str
    text_box_out_file_ax = fig_han.add_axes([0.72, 0.09, 0.27, 0.03])                # [left bottom fraction_horizontal  fraction_vertical ]
    text_box_out_file = TextBox(text_box_out_file_ax, 'output filename:', initial=output_filename)
    text_box_out_file.text_disp.set_fontsize(7)
    text_box_cmap.on_submit(output_filename_textbox_update)


    # Create an interpolator so we can quickly get the value of TA_data at any wavelength or delay_time, and then we print the value of TA_data in the window bar
    TA_matrix_interpolator = RegularGridInterpolator( (TA_matrix_delay_times, np.flip(TA_matrix_wavelengths)), np.fliplr(TA_data), bounds_error=False, fill_value=None)
    TA_image_axis_han.format_coord = lambda x, y: f"crosshairs (w: {TA_matrix_wavelengths[crosshair_v_idx]:0.1f} nm, t: {TA_matrix_delay_times[crosshair_h_idx]:0.1f} ps, z: {TA_data[crosshair_h_idx,crosshair_v_idx]:0.6f})           mouse pointer (w: {x:.1f}, t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f})   "
    ax_horiz.format_coord          = lambda x, y: f"mouse pointer location (w: {x:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"
    ax_vert.format_coord           = lambda x, y: f"mouse pointer location (t: {y:.1f}, z: {TA_matrix_interpolator([y, x])[0]:.6f}"


    #### MOUSE Click Handler Function
    def on_mouse_click(event):
        nonlocal crosshair_v_idx
        nonlocal crosshair_h_idx
        # Check if click is inside the image axis AND is the Left Button AND is a double-click (We want this to move the crosshairs!)
        if event.inaxes == TA_image_axis_han and event.dblclick and (event.button == 1 or event.button == 3):
            
            # Where was the mouse clicked?
            x_mouse, y_mouse = event.xdata, event.ydata     #x_mouse and y_mouse here are in axis units of wavelength and delay_time
            
            # If the r button was pushed, then record this location
            if event.button == 3 and x_mouse > click_history[-1][0] :
                click_history.extend([[x_mouse, y_mouse]])

            # Update Crosshairs to the clicked location
            crosshair_v.set_xdata([x_mouse])
            crosshair_h.set_ydata([y_mouse])

            # Convert to integer indices
            crosshair_v_idx = int(np.abs(TA_matrix_wavelengths - x_mouse).argmin())
            crosshair_h_idx = int(np.abs(TA_matrix_delay_times - y_mouse).argmin())

            # Update the window toolbar message
            if isinstance(event.xdata,int):
                fig_han.canvas.toolbar.set_message( TA_image_axis_han.format_coord(event.xdata, event.ydata) )

            # Obtain the TA_data only in the zoomed view
            xlim = TA_image_axis_han.get_xlim()
            ylim = TA_image_axis_han.get_ylim()
            zoom_w_idx = [ int(np.abs(TA_matrix_wavelengths - xlim[0]).argmin()), int(np.abs(TA_matrix_wavelengths - xlim[1]).argmin()) ]
            zoom_t_idx = [ int(np.abs(TA_matrix_delay_times - ylim[0]).argmin()), int(np.abs(TA_matrix_delay_times - ylim[1]).argmin()) ]

            # Update Horizontal Transect (Bottom Plot)
            # Fetch the column at crosshair_v_idx
            line_horiz.set_ydata(TA_data[crosshair_h_idx, :]) 
            ax_horiz.set_ylim([TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].min(), TA_data[crosshair_h_idx,min(zoom_w_idx):max(zoom_w_idx)].max()])
            red_dot_horiz.set_offsets([TA_matrix_wavelengths[crosshair_v_idx], TA_data[crosshair_h_idx,crosshair_v_idx]])

            # Update Vertical Transect (Side Plot)
            # Fetch the row at crosshair_h_idx
            line_vert.set_xdata(TA_data[:, crosshair_v_idx])
            ax_vert.set_xlim([TA_data[:,crosshair_v_idx].min(), TA_data[:,crosshair_v_idx].max()])    
            red_dot_vert.set_offsets([TA_data[crosshair_h_idx,crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]])

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
        if (event.key == 'up' or event.key == 'down' or event.key == 'left' or event.key == 'right' ):
            # Update Crosshairs to the clicked location
            crosshair_h.set_ydata([TA_matrix_delay_times[crosshair_h_idx], TA_matrix_delay_times[crosshair_h_idx]])
            crosshair_v.set_xdata([TA_matrix_wavelengths[crosshair_v_idx], TA_matrix_wavelengths[crosshair_v_idx]])
            # Update the window toolbar message
            if isinstance(event.xdata,int):
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

    # A button to save the crosshair location for t0 fit line
    def save_crosshairs(event):
        if TA_matrix_wavelengths[crosshair_v_idx] > click_history[-1][0] :         # check if the user clicked on the same spot twice or clicked on a smaller wavelength
            click_history.extend([[TA_matrix_wavelengths[crosshair_v_idx], TA_matrix_delay_times[crosshair_h_idx]]])
    # Create a button that saves the crosshairs to a history of crosshairs
    ax_button_save_crosshairs = plt.axes([0.6, 0.18, 0.11, 0.03])  # [left, bottom, width, height]
    Button_sv_crosshairs = Button(ax_button_save_crosshairs, 'Save Crosshairs')
    #Button_sv_crosshairs.text_disp.set_fontsize(7)
    Button_sv_crosshairs.on_clicked(save_crosshairs)  


    # A button to create a fit line to the t0 locations
    def fit_t0(event):
        # Create spline fit of clicked points
        np_click_history = np.array(click_history[1:])     # cut away the first click_history data because it was artificial
        t0_spline_interpolator = CubicSpline(np_click_history[:,0], np_click_history[:,1], extrapolate = True)
        t0_each_wavelength[:] = t0_spline_interpolator(TA_matrix_wavelengths)
        t0_fit_line.set_data(TA_matrix_wavelengths , t0_each_wavelength)
        t0_fit_line.set_alpha(0.8)
        # Redraw
        fig_han.canvas.draw_idle()
    ax_button_t0_fit = plt.axes([0.725, 0.18, 0.07, 0.03])  # [left, bottom, width, height]
    Button_t0 = Button(ax_button_t0_fit, 'fit t0')
    Button_t0.on_clicked(fit_t0)    



    # A button to correct the TA data for t0 chirp
    def correct_t0_background(event):
        # Create an interpolation with t0 values shifting t values downward
        grid_delay_times             = np.outer(TA_matrix_delay_times, np.ones(len(TA_matrix_wavelengths)) )  
        scatter_delay_times_minus_t0 = np.outer(TA_matrix_delay_times, np.ones(len(TA_matrix_wavelengths)) ) - t0_each_wavelength 
        grid_wavelengths             = np.outer(np.ones(len(TA_matrix_delay_times)), TA_matrix_wavelengths)
        # Pad the TA_data in the time delay directions so the interpolation doesn't produce NaNs
        TA_data_padded = np.vstack((TA_data[0,:], TA_data))
        TA_data_padded = np.vstack((TA_data_padded, TA_data_padded[-1,:]))
        scatter_delay_times_minus_t0 = np.vstack((scatter_delay_times_minus_t0[0,:]-1.0, scatter_delay_times_minus_t0))
        scatter_delay_times_minus_t0 = np.vstack((scatter_delay_times_minus_t0, scatter_delay_times_minus_t0[-1,:]+50))
        grid_wavelengths_padded = np.vstack((grid_wavelengths[0,:], grid_wavelengths))
        grid_wavelengths_padded = np.vstack((grid_wavelengths_padded, grid_wavelengths_padded[-1,:]))
        TA_data[:] = griddata( (scatter_delay_times_minus_t0.ravel(), grid_wavelengths_padded.ravel()), TA_data_padded.ravel(), (grid_delay_times, grid_wavelengths), method='linear')  # Options: 'nearest', 'linear', 'cubic'
        # Remove the t0 fit line
        t0_fit_line.set_alpha(0.0)
        # Count the NaN and report 
        print(str(np.sum(np.isnan(TA_data))) + ' NaN found in the interpolated array. They were set to 0.')
        TA_data[np.isnan(TA_data)] = 0
        # Remove the background counts (scattered light)
        background_each_wavelength = np.zeros(len(TA_matrix_wavelengths))
        t0_idx = int(np.abs(TA_matrix_delay_times).argmin())
        st_idx = int(TA_matrix_delay_times.argmin())
        print(t0_idx, st_idx)
        for repeat in range(0,30):
            for j in range(1, len(TA_matrix_wavelengths)-2):
                if st_idx < t0_idx:
                    background_each_wavelength[j] = np.sum(TA_data[st_idx:t0_idx-15, j-1:j+1])/(t0_idx - st_idx)/3
                    background_each_wavelength[0] = np.sum(TA_data[st_idx:t0_idx-15, 0])/(t0_idx - st_idx)
                    background_each_wavelength[-1] = np.sum(TA_data[st_idx:t0_idx-15, -1])/(t0_idx - st_idx)
                elif st_idx > t0_idx: 
                    background_each_wavelength[j] = np.sum(TA_data[t0_idx:st_idx+15, j-1:j+1])/(st_idx - t0_idx)/3
                    background_each_wavelength[0] = np.sum(TA_data[t0_idx:st_idx+15, 0])/(st_idx - t0_idx)
                    background_each_wavelength[-1] = np.sum(TA_data[t0_idx:st_idx+15, -1])/(st_idx - t0_idx)
            TA_data[:] = TA_data - background_each_wavelength
        # Update the TA image plot
        pclrmsh.set_array(TA_data.ravel())
        # Recreate an interpolator so we can quickly get the value of TA_data at any wavelength or delay_time, and then we print the value of TA_data in the window bar
        TA_matrix_interpolator = RegularGridInterpolator( (TA_matrix_delay_times, TA_matrix_wavelengths), TA_data, bounds_error=False, fill_value=None)
        TA_image_axis_han.format_coord = lambda x, y: f"ch w:{TA_matrix_wavelengths[crosshair_v_idx]:0.1f} nm, \tch t:{TA_matrix_delay_times[crosshair_h_idx]:0.1f} ps \t\t\t\t\t\t\t\t\tW: {x:.1f}, \tT: {y:.1f}, \tZ: {TA_matrix_interpolator([y, x])[0]:.6f}     ."

        # Redraw
        fig_han.canvas.draw_idle()
    # Define the position and size of the button that corrects chirp of t0 and removes background counts
    ax_button_t0_correct = plt.axes([0.81, 0.18, 0.17, 0.03])  # [left, bottom, width, height]
    Button_t0_correct = Button(ax_button_t0_correct, 'correct t0 & background')
    Button_t0_correct.on_clicked(correct_t0_background)


    def save_TA_to_disk(event):
        TA_matrix[1:,1:] = TA_data
        np.savetxt(output_filename, TA_matrix, delimiter=',')
    # Create a button to saves the TA_matrix to disk
    ax_button_save_TA_to_disk = plt.axes([0.82, 0.13, 0.12, 0.03])  # [left, bottom, width, height]
    Button_save_TA_to_disk = Button(ax_button_save_TA_to_disk, 'Save TA to disk')
    Button_save_TA_to_disk.on_clicked(save_TA_to_disk)  


    # Event handler for mouse clicking
    fig_han.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Event handler for up down buttons
    fig_han.canvas.mpl_connect('key_press_event', on_up_down_key_press)

    # Event handler to check if the mouse is in the plot window
    fig_han.canvas.mpl_connect('motion_notify_event', update_mouse_in_plot_axes)

    plt.show(block=True)        # this is needed to stop the program from executing the return immediately









