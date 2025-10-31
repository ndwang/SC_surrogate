import os
import sys
import argparse
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def iter_efield_values(h5_file: h5py.File, sample_keys: list, component: str = 'all'):
    """
    Yield numpy arrays of raw electric field values for each sample.

    Raw layout per sample in the raw HDF5 is expected to be (Nx, Ny, Nz, 3).
    component: 'all' | 'Ex' | 'Ey' | 'Ez'
    """
    comp_map = {'Ex': 0, 'Ey': 1, 'Ez': 2}
    comp_idx = None if component == 'all' else comp_map[component]
    for key in sample_keys:
        grp = h5_file[key]
        efield = grp['efield'][:]  # shape (Nx, Ny, Nz, 3)
        if comp_idx is None:
            yield efield.reshape(-1)
        else:
            yield efield[..., comp_idx].reshape(-1)


def compute_abs_min_max(h5_path: str, sample_keys: list, component: str) -> dict:
    """Compute streaming stats on absolute values: count, min, max."""
    count = 0
    vmin = np.inf
    vmax = -np.inf
    with h5py.File(h5_path, 'r') as f:
        for arr in iter_efield_values(f, sample_keys, component):
            if arr.size == 0:
                continue
            arr = np.abs(arr)
            local_min = float(np.min(arr))
            local_max = float(np.max(arr))
            if local_min < vmin:
                vmin = local_min
            if local_max > vmax:
                vmax = local_max
            count += int(arr.size)
    if count == 0:
        raise RuntimeError('No values found to compute statistics.')
    return {
        'count': count,
        'min': float(vmin),
        'max': float(vmax),
    }


def compute_histogram(h5_path: str, sample_keys: list, bins: int, value_range: tuple[float, float], component: str) -> tuple[np.ndarray, np.ndarray]:
    counts = None
    bin_edges = None
    with h5py.File(h5_path, 'r') as f:
        for arr in iter_efield_values(f, sample_keys, component):
            abs_arr = np.abs(arr)
            c, be = np.histogram(abs_arr, bins=bins, range=value_range)
            if counts is None:
                counts = c.astype(np.int64)
                bin_edges = be
            else:
                counts += c
    return counts, bin_edges


def median_from_histogram(counts: np.ndarray, bin_edges: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total == 0:
        return float('nan')
    half = (total + 1) // 2  # median rank (1-based), works for even/odd
    cum = 0
    for i, c in enumerate(counts):
        next_cum = cum + int(c)
        if next_cum >= half:
            # Interpolate within bin assuming uniform distribution
            bin_left = bin_edges[i]
            bin_right = bin_edges[i + 1]
            in_bin_rank = half - cum
            frac = in_bin_rank / max(int(c), 1)
            return float(bin_left + frac * (bin_right - bin_left))
        cum = next_cum
    return float(bin_edges[-1])


def main():
    parser = argparse.ArgumentParser(description='Plot histogram of raw electric field values from raw HDF5 data.')
    parser.add_argument('raw_path', nargs='?', default=None, help='Path to raw HDF5 file (group-per-sample). If omitted, falls back to config')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training config YAML to locate raw_data_path')
    parser.add_argument('--bins', type=int, default=200, help='Number of histogram bins')
    parser.add_argument('--range-min', type=float, default=None, help='Minimum value for histogram range')
    parser.add_argument('--range-max', type=float, default=None, help='Maximum value for histogram range')
    parser.add_argument('--component', type=str, choices=['all', 'Ex', 'Ey', 'Ez'], default='all', help='Which component to include')
    parser.add_argument('--save', type=str, default=None, help='Path to save the plot (PNG). If not set, display interactively')
    args = parser.parse_args()

    # Resolve raw file path via positional -> config (following preprocess_data.py approach)
    if args.raw_path is not None:
        h5_path = args.raw_path
    else:
        cfg = load_config(args.config)
        try:
            h5_path = cfg['paths']['raw_data_path']
        except Exception as e:
            print(f"Error: could not read paths.raw_data_path from config: {e}")
            sys.exit(1)

    if not os.path.isfile(h5_path):
        print(f"Error: raw file not found: {h5_path}")
        sys.exit(1)

    # Collect sample keys (same as Preprocessor.examine_raw_data)
    with h5py.File(h5_path, 'r') as f:
        sample_keys = list(f.keys())
        if not sample_keys:
            print('Error: no samples found in raw HDF5 file.')
            sys.exit(1)

    # First pass: abs min/max and count
    stats = compute_abs_min_max(h5_path, sample_keys, args.component)

    # Determine histogram range
    if args.range_min is None or args.range_max is None:
        vmin, vmax = stats['min'], stats['max']
    else:
        vmin, vmax = float(args.range_min), float(args.range_max)
        if vmin >= vmax:
            print('Error: --range-min must be less than --range-max')
            sys.exit(1)

    # Second pass: histogram on |E|
    counts, bin_edges = compute_histogram(h5_path, sample_keys, args.bins, (vmin, vmax), args.component)

    # Median from histogram
    median_val = median_from_histogram(counts, bin_edges)

    print(
        f"|E| statistics ({'all components' if args.component=='all' else args.component}):\n"
        f"  min:    {stats['min']:.6g}\n"
        f"  max:    {stats['max']:.6g}\n"
        f"  median: {median_val:.6g}"
    )

    # Plot
    plt.figure(figsize=(9, 5))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[:, None][1:].ravel())
    plt.bar(centers, counts, width=(bin_edges[1] - bin_edges[0]), align='center', alpha=0.7, edgecolor='black')
    title_comp = 'All components' if args.component == 'all' else args.component
    plt.title(f'Histogram of |E| (Raw Values) ({title_comp})')
    plt.xlabel('|E| (raw)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f'Saved histogram to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()


