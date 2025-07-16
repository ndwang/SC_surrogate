import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import argparse
from evaluation.visualization_tools.raw_data import plot_density, plot_efield, plot_both, get_run_data

def main():
    parser = argparse.ArgumentParser(description='Visualize raw charge density and electric field data from HDF5 file.')
    parser.add_argument('input_file', type=str, help='Path to HDF5 file containing space charge data')
    parser.add_argument('--plot', type=str, choices=['density', 'efield', 'both'], default='density', help='What to plot: density, efield, or both')
    parser.add_argument('--run', type=str, required=True, help='Specific run to visualize (e.g., "run_00000")')
    args = parser.parse_args()

    ext = os.path.splitext(args.input_file)[1].lower()
    if ext not in ['.h5', '.hdf5']:
        print('Error: Only .h5 and .hdf5 files are supported.')
        sys.exit(1)

    try:
        with h5py.File(args.input_file, 'r') as h5_file:
            if args.run not in h5_file:
                print(f"Error: Run '{args.run}' not found.")
                sys.exit(1)
            parameters, rho, efield = get_run_data(h5_file, args.run)
            run_info = f"({args.run})"
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