import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import argparse
from evaluation.visualization_tools.raw_data import plot_density, plot_efield, plot_both, get_run_data
from evaluation.visualization_tools.training_curves import plot_training_curves
from evaluation.visualization_tools.predict_efield import predict_and_visualize_efield

def main():
    parser = argparse.ArgumentParser(description='Visualization CLI for raw data, training curves, and model predictions.')
    parser.add_argument('input_file', type=str, nargs='?', default=None, help='Path to HDF5 file, training history pickle file, or processed test set')
    parser.add_argument('--plot', type=str, choices=['density', 'efield', 'both', 'curves', 'predict_efield'], default='density', 
                       help='What to plot: density (rho), efield, both, training curves, or predict efield')
    parser.add_argument('--run', type=str, default=None, help='Specific run to visualize (e.g., "run_00000")')
    parser.add_argument('--checkpoint', type=str, default='saved_models/best_model.pth', help='Path to model checkpoint for prediction')
    parser.add_argument('--scalers', type=str, default='saved_models/scalers.pkl', help='Path to scalers.pkl for denormalization')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training config')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index from processed test set to visualize')
    args = parser.parse_args()

    if args.plot == 'curves':
        if args.input_file is None:
            print('Error: Please provide the path to training_history.pkl as input_file.')
            sys.exit(1)
        plot_training_curves(args.input_file)
        return

    if args.plot == 'predict_efield':
        if args.input_file is None:
            print('Error: Please provide the path to the processed test set as input_file.')
            sys.exit(1)
        predict_and_visualize_efield(args.config, args.checkpoint, args.scalers, args.input_file, args.sample_idx)
        return

    if args.input_file is None:
        print('Error: Please provide the path to the HDF5 file as input_file.')
        sys.exit(1)

    ext = os.path.splitext(args.input_file)[1].lower()
    if ext not in ['.h5', '.hdf5']:
        print('Error: Only .h5 and .hdf5 files are supported for density/efield visualization.')
        sys.exit(1)

    if args.run is None:
        print('Error: Please specify --run for density/efield visualization.')
        sys.exit(1)

    try:
        with h5py.File(args.input_file, 'r') as h5_file:
            if args.run not in h5_file:
                print(f"Error: Run '{args.run}' not found.")
                sys.exit(1)
            selected_run = args.run
            parameters, rho, efield = get_run_data(h5_file, selected_run)
            run_info = f"({selected_run})"
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