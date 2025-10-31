import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml
import numpy as np
from modeling.models import create_model_from_config

def plot_kernel_slices_with_weights(kernel, title=None):
    # kernel: (out_channels, in_channels, D, H, W)
    out_ch, in_ch, D, H, W = kernel.shape
    oc_init = 0
    ic_init = 0

    def plot_slices(oc, ic):
        k = kernel[oc, ic]  # (D, H, W)
        fig.suptitle(title if title else "")
        for i in range(3):
            axes[i].clear()
            if i >= k.shape[0]:
                axes[i].set_visible(False)
                continue
            axes[i].imshow(k[i, :, :], cmap='viridis')
            axes[i].set_title(f'Slice [:,:,{i}] (oc={oc}, ic={ic})')
            # Overlay weights as text
            for y in range(k.shape[1]):
                for x in range(k.shape[2]):
                    val = k[i, y, x]
                    axes[i].text(x, y, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold',
                                 bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        fig.canvas.draw_idle()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)
    plot_slices(oc_init, ic_init)

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_oc = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_ic = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    slider_oc = Slider(ax_oc, 'Output Channel', 0, out_ch-1, valinit=oc_init, valstep=1)
    slider_ic = Slider(ax_ic, 'Input Channel', 0, in_ch-1, valinit=ic_init, valstep=1)

    def update(val):
        oc = int(slider_oc.val)
        ic = int(slider_ic.val)
        plot_slices(oc, ic)

    slider_oc.on_changed(update)
    slider_ic.on_changed(update)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize CNN3D convolution kernels from a checkpoint.')
    parser.add_argument('--checkpoint', type=str, default='saved_models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training config')
    parser.add_argument('--layer_type', type=str, choices=['encoder', 'decoder'], default='encoder', help='Layer type to visualize')
    parser.add_argument('--layer_idx', type=int, default=0, help='Index of the layer to visualize')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Error: File '{args.checkpoint}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: File '{args.config}' not found.")
        sys.exit(1)

    # Load config and checkpoint
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = create_model_from_config(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Select layer
    if args.layer_type == 'encoder':
        layers = model.encoder_layers
    else:
        layers = model.decoder_layers
    if args.layer_idx < 0 or args.layer_idx >= len(layers):
        print(f"Error: {args.layer_type} layer index {args.layer_idx} out of range (0-{len(layers)-1})")
        sys.exit(1)
    layer = layers[args.layer_idx]
    if not isinstance(layer, torch.nn.Conv3d):
        print(f"Error: Selected layer is not Conv3d.")
        sys.exit(1)
    kernel = layer.weight.data.cpu().numpy()
    plot_kernel_slices_with_weights(kernel, title=f'{args.layer_type.capitalize()} Layer {args.layer_idx} Kernels')

if __name__ == '__main__':
    main() 