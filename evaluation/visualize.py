import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import sys
import os
import h5py
from functools import partial

def get_slice_nd(arr, axis, index, comp=None):
    # arr: ndarray, axis: int, index: int, comp: int or None
    # For 3D: comp=None, for 4D: comp=component index
    if comp is None:
        if axis == 0:
            return arr[index, :, :]
        elif axis == 1:
            return arr[:, index, :]
        elif axis == 2:
            return arr[:, :, index]
    else:
        if axis == 0:
            return arr[index, :, :, comp]
        elif axis == 1:
            return arr[:, index, :, comp]
        elif axis == 2:
            return arr[:, :, index, comp]

def interactive_slice_plot(plot_specs, shape, get_slice_funcs, window_title=None):
    nplots = len(plot_specs)
    init_axis = 0
    init_index = shape[init_axis] // 2

    fig, axs = plt.subplots(1, nplots, figsize=(6*nplots, 5))
    if nplots == 1:
        axs = [axs]
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.98)
    if window_title:
        fig.canvas.manager.set_window_title(window_title)

    imgs = []
    for i, ax in enumerate(axs):
        arr2d = get_slice_funcs[i](init_axis, init_index)
        img = ax.imshow(arr2d, origin='lower', cmap=plot_specs[i]['cmap'], vmin=plot_specs[i]['vmin'], vmax=plot_specs[i]['vmax'])
        ax.set_title(f"{plot_specs[i]['title']} | Axis: {init_axis}, Slice: {init_index}")
        plt.colorbar(img, ax=ax)
        imgs.append(img)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=shape[init_axis] - 1,
        valinit=init_index,
        valstep=1,
    )

    ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('0', '1', '2'), active=init_axis)

    def update(val):
        axis = int(radio.value_selected)
        index = int(slider.val)
        for i, img in enumerate(imgs):
            arr2d = get_slice_funcs[i](axis, index)
            img.set_data(arr2d)
            img.set_clim(plot_specs[i]['vmin'], plot_specs[i]['vmax'])
            axs[i].set_title(f"{plot_specs[i]['title']} | Axis: {axis}, Slice: {index}")
        fig.canvas.draw_idle()

    def update_axis(label):
        axis = int(label)
        slider.valmax = shape[axis] - 1
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.set_val(shape[axis] // 2)
        update(None)

    slider.on_changed(update)
    radio.on_clicked(update_axis)
    plt.show()

def plot_density(rho, run_info=""):
    vmin = np.min(rho)
    vmax = np.max(rho)
    get_slice = partial(get_slice_nd, rho, comp=None)
    plot_specs = [{
        'title': f'Density {run_info}',
        'cmap': 'viridis',
        'vmin': vmin,
        'vmax': vmax,
    }]
    interactive_slice_plot(plot_specs, rho.shape, [get_slice], window_title=f'Density {run_info}')

def plot_efield(efield, run_info=""):
    if efield.ndim != 4 or efield.shape[3] != 3:
        print('Error: efield must have shape (Nx, Ny, Nz, 3)')
        sys.exit(1)
    vmin = np.min(efield)
    vmax = np.max(efield)
    component_names = ['Ex', 'Ey', 'Ez']
    get_slice_funcs = [partial(get_slice_nd, efield, comp=i) for i in range(3)]
    plot_specs = [
        {'title': f'{name} {run_info}', 'cmap': 'RdBu', 'vmin': vmin, 'vmax': vmax}
        for name in component_names
    ]
    interactive_slice_plot(plot_specs, efield.shape[:3], get_slice_funcs, window_title=f'Efield {run_info}')

def plot_both(rho, efield, run_info=""):
    if rho.shape != efield.shape[:3]:
        print('Error: rho and efield spatial dimensions do not match.')
        sys.exit(1)
    vmin_rho = np.min(rho)
    vmax_rho = np.max(rho)
    vmin_ef = np.min(efield)
    vmax_ef = np.max(efield)
    component_names = ['Ex', 'Ey', 'Ez']
    get_slice_rho = partial(get_slice_nd, rho, comp=None)
    get_slice_funcs = [get_slice_rho] + [partial(get_slice_nd, efield, comp=i) for i in range(3)]
    plot_specs = [
        {'title': f'Density {run_info}', 'cmap': 'viridis', 'vmin': vmin_rho, 'vmax': vmax_rho}
    ] + [
        {'title': f'{name} {run_info}', 'cmap': 'RdBu', 'vmin': vmin_ef, 'vmax': vmax_ef}
        for name in component_names
    ]
    interactive_slice_plot(plot_specs, rho.shape, get_slice_funcs, window_title=f'Density + Efield {run_info}')

def list_available_runs(h5_file, max_display=20):
    """List available runs in the HDF5 file with a safety limit."""
    runs = []
    for key in h5_file.keys():
        if key.startswith('run_'):
            runs.append(key)
    runs = sorted(runs)
    
    if len(runs) > max_display:
        print(f"Warning: Found {len(runs)} runs, showing only first {max_display}")
        print("Use --run <run_name> to specify a specific run")
        return runs[:max_display]
    return runs

def get_run_data(h5_file, run_name):
    """Extract data from a specific run."""
    run_group = h5_file[run_name]
    parameters = run_group['parameters'][:]
    rho = run_group['rho'][:]
    efield = run_group['efield'][:]
    return parameters, rho, efield

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize 3D charge density or electric field from HDF5 file with interactive slicing.')
    parser.add_argument('input_file', type=str, help='Path to HDF5 file containing space charge data')
    parser.add_argument('--plot', type=str, choices=['density', 'efield', 'both'], default='density', 
                       help='What to plot: density (rho), efield, or both')
    parser.add_argument('--run', type=str, required=True, help='Specific run to visualize (e.g., "run_00000")')
    args = parser.parse_args()

    ext = os.path.splitext(args.input_file)[1].lower()
    if ext not in ['.h5', '.hdf5']:
        print('Error: Only .h5 and .hdf5 files are supported.')
        sys.exit(1)

    try:
        with h5py.File(args.input_file, 'r') as h5_file:
            # Check if the run exists directly in HDF5 (fast)
            if args.run not in h5_file:
                print(f"Error: Run '{args.run}' not found.")
                sys.exit(1)
            selected_run = args.run
            
            # Load data for the selected run
            parameters, rho, efield = get_run_data(h5_file, selected_run)
            
            # Create run info string for plot titles
            run_info = f"({selected_run})"
            
            # Plot based on user selection
            if args.plot == 'density':
                if rho.ndim != 3:
                    print('Error: rho must be a 3D array.')
                    sys.exit(1)
                plot_density(rho, run_info)
            elif args.plot == 'efield':
                plot_efield(efield, run_info)
            elif args.plot == 'both':
                plot_both(rho, efield, run_info)
                
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 