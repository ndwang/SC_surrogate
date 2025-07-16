import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import joblib
import torch
from modeling.dataset import SpaceChargeDataset
import matplotlib.pyplot as plt
import yaml
from evaluation.visualization_tools.raw_data import interactive_slice_plot, get_slice_nd
import numpy as np

def plot_efield_vs_truth(pred, truth, sample_idx):
    # pred, truth: (3, Nx, Ny, Nz)
    # Show Ex, Ey, Ez for both pred and truth, middle slice
    component_names = ['Ex', 'Ey', 'Ez']
    ncomp = 3
    z_slice = pred.shape[-1] // 2
    fig, axes = plt.subplots(2, ncomp, figsize=(5*ncomp, 8))
    for i, comp in enumerate(component_names):
        axes[0, i].imshow(pred[i, :, :, z_slice], cmap='RdBu')
        axes[0, i].set_title(f'Predicted {comp} (sample {sample_idx})')
        axes[1, i].imshow(truth[i, :, :, z_slice], cmap='RdBu')
        axes[1, i].set_title(f'True {comp} (sample {sample_idx})')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict and visualize electric field from a trained model checkpoint, with two modes: ef (efield vs ground truth) and rho (charge density vs prediction).')
    parser.add_argument('test_file', type=str, help='Path to processed test set (e.g., data/processed/test.h5)')
    parser.add_argument('--checkpoint', type=str, default='saved_models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--scalers', type=str, default='saved_models/scalers.pkl', help='Path to scalers.pkl for denormalization')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training config')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index from processed test set to visualize')
    parser.add_argument('--mode', type=str, choices=['compare', 'predict'], default='compare',
                        help="'compare': plot predicted efield vs ground truth; 'predict': plot charge density vs predicted efield")
    args = parser.parse_args()

    if not os.path.isfile(args.test_file):
        print(f"Error: File '{args.test_file}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"Error: File '{args.checkpoint}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.scalers):
        print(f"Error: File '{args.scalers}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: File '{args.config}' not found.")
        sys.exit(1)

    # Load scalers
    try:
        scalers = joblib.load(args.scalers)
        input_scaler = scalers['input_scaler']
        target_scaler = scalers['target_scaler']
    except Exception:
        input_scaler = None
        target_scaler = None
        print("Warning: Could not load scalers, will plot normalized data.")

    # Load sample
    dataset = SpaceChargeDataset(args.test_file, device='cpu')
    input_tensor, target_tensor = dataset[args.sample_idx]
    # input_tensor: (1, Nx, Ny, Nz), target_tensor: (3, Nx, Ny, Nz)
    input_np = input_tensor.squeeze(0).numpy()
    target_np = target_tensor.numpy()
    # Denormalize
    if input_scaler is not None:
        input_flat = input_np.reshape(-1, 1)
        input_denorm = input_scaler.inverse_transform(input_flat).reshape(input_np.shape)
    else:
        input_denorm = input_np
    if target_scaler is not None:
        target_flat = target_np.reshape(3, -1).T
        target_denorm = target_scaler.inverse_transform(target_flat).T.reshape(target_np.shape)
    else:
        target_denorm = target_np
    # Predict
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    from modeling.models import create_model_from_config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model_from_config(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    input_for_pred = torch.tensor(input_tensor).unsqueeze(0).to(device) if input_tensor.ndim == 4 else input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_for_pred).cpu().numpy()[0]  # (3, Nx, Ny, Nz)
    if target_scaler is not None:
        pred_flat = pred.reshape(3, -1).T
        pred_denorm = target_scaler.inverse_transform(pred_flat).T.reshape(pred.shape)
    else:
        pred_denorm = pred

    if args.mode == 'compare':
        # Prepare predicted and true efield for interactive plotting
        # Convert to (Nx, Ny, Nz, 3) if needed
        def to_channel_last(arr):
            if arr.ndim == 4:
                if arr.shape[0] == 3 and arr.shape[-1] != 3:
                    return np.transpose(arr, (1, 2, 3, 0))
                elif arr.shape[-1] == 3:
                    return arr
                else:
                    raise ValueError(f"efield shape {arr.shape} is not compatible (expected (Nx, Ny, Nz, 3) or (3, Nx, Ny, Nz))")
            else:
                raise ValueError(f"efield must be 4D, got shape {arr.shape}")
        pred_efield = to_channel_last(pred_denorm)
        true_efield = to_channel_last(target_denorm)
        # Use same vmin/vmax for each component across pred/true
        component_names = ['Ex', 'Ey', 'Ez']
        from functools import partial
        get_slice_funcs = []
        plot_specs = []
        # Prediction panels (first row)
        for i, name in enumerate(component_names):
            vmin = pred_efield[..., i].min()
            vmax = pred_efield[..., i].max()
            get_slice_funcs.append(partial(get_slice_nd, pred_efield, comp=i))
            plot_specs.append({'title': f'Predicted {name}', 'cmap': 'RdBu', 'vmin': vmin, 'vmax': vmax})
        # Ground truth panels (second row)
        for i, name in enumerate(component_names):
            vmin = true_efield[..., i].min()
            vmax = true_efield[..., i].max()
            get_slice_funcs.append(partial(get_slice_nd, true_efield, comp=i))
            plot_specs.append({'title': f'True {name}', 'cmap': 'RdBu', 'vmin': vmin, 'vmax': vmax})
        interactive_slice_plot(plot_specs, pred_efield.shape[:3], get_slice_funcs, window_title=f'Sample {args.sample_idx}: Predicted vs True E-field', row_labels=['Prediction', 'Ground Truth'])
    elif args.mode == 'predict':
        # Prepare density and predicted efield for plotting
        # Convert pred_denorm to (Nx, Ny, Nz, 3) if needed
        if pred_denorm.ndim == 4:
            if pred_denorm.shape[0] == 3 and pred_denorm.shape[-1] != 3:
                # (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3)
                pred_efield = np.transpose(pred_denorm, (1, 2, 3, 0))
            elif pred_denorm.shape[-1] == 3:
                pred_efield = pred_denorm
            else:
                raise ValueError(f"pred_denorm shape {pred_denorm.shape} is not compatible (expected (Nx, Ny, Nz, 3) or (3, Nx, Ny, Nz))")
        else:
            raise ValueError(f"pred_denorm must be 4D, got shape {pred_denorm.shape}")
        # Squeeze density if it has a leading singleton or channel dimension
        if input_denorm.ndim == 4 and input_denorm.shape[0] == 1:
            density = np.squeeze(input_denorm, axis=0)
        elif input_denorm.ndim == 4 and input_denorm.shape[0] != pred_efield.shape[-1]:
            density = np.squeeze(input_denorm)
        else:
            density = input_denorm
        if density.ndim != 3:
            raise ValueError(f"density shape {density.shape} is not compatible (expected (Nx, Ny, Nz))")
        # Prepare plot specs and slice functions as in plot_both
        vmin_rho = density.min()
        vmax_rho = density.max()
        vmin_ef = pred_efield.min()
        vmax_ef = pred_efield.max()
        component_names = ['Ex', 'Ey', 'Ez']
        from functools import partial
        get_slice_density = partial(get_slice_nd, density, comp=None)
        get_slice_funcs = [get_slice_density] + [partial(get_slice_nd, pred_efield, comp=i) for i in range(3)]
        plot_specs = [
            {'title': 'Charge Density', 'cmap': 'viridis', 'vmin': vmin_rho, 'vmax': vmax_rho}
        ] + [
            {'title': f'Predicted {name}', 'cmap': 'RdBu', 'vmin': vmin_ef, 'vmax': vmax_ef}
            for name in component_names
        ]
        interactive_slice_plot(plot_specs, density.shape, get_slice_funcs, window_title=f'Sample {args.sample_idx}: Density + Predicted E-field')

if __name__ == '__main__':
    main() 