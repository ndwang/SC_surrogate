"""
Data preprocessing script for Space Charge Surrogate Model.

This script converts raw simulation data from group-per-sample HDF5 format 
to monolithic format, applies normalization, and splits into train/val/test sets.

Key features:
- Converts from group-per-sample to efficient monolithic format
- Fits StandardScaler ONLY on training data (prevents data leakage)
- Handles correct tensor shapes for PyTorch CNN compatibility
- Memory-efficient processing with proper error handling
"""

import os
import h5py
import numpy as np
import yaml
import joblib
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def examine_raw_data(raw_data_path: str) -> Tuple[List[str], Tuple[int, int, int]]:
    """
    Examine raw HDF5 file to get sample keys and data dimensions.
    
    Args:
        raw_data_path: Path to raw HDF5 file
        
    Returns:
        Tuple of (sample_keys, grid_shape)
    """
    logger.info(f"Examining raw data file: {raw_data_path}")
    
    with h5py.File(raw_data_path, 'r') as f:
        sample_keys = list(f.keys())
        
        # Get grid dimensions from first sample
        first_key = sample_keys[0]
        rho_shape = f[first_key]['rho'].shape
        efield_shape = f[first_key]['efield'].shape
        
        logger.info(f"Found {len(sample_keys)} samples")
        logger.info(f"rho shape: {rho_shape}, efield shape: {efield_shape}")
        
        # Validate that efield needs transposition
        if efield_shape[-1] == 3:
            logger.info("efield has shape (Nx,Ny,Nz,3) - will transpose to (3,Nx,Ny,Nz)")
        
        return sample_keys, rho_shape


def split_indices(n_samples: int, split_ratios: List[float], seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Split sample indices into train, validation, and test sets.
    
    Args:
        n_samples: Total number of samples
        split_ratios: [train_ratio, val_ratio, test_ratio]
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(split_ratios)}"
    
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * split_ratios[0])
    val_size = int(n_samples * split_ratios[1])
    
    train_indices = shuffled_indices[:train_size].tolist()
    val_indices = shuffled_indices[train_size:train_size + val_size].tolist()
    test_indices = shuffled_indices[train_size + val_size:].tolist()
    
    logger.info(f"Data split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def fit_scalers(raw_data_path: str, train_indices: List[int], sample_keys: List[str]) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit StandardScaler objects on training data ONLY.
    
    Args:
        raw_data_path: Path to raw HDF5 file
        train_indices: Indices of training samples
        sample_keys: List of all sample keys
        
    Returns:
        Tuple of (input_scaler, target_scaler)
    """
    logger.info("Fitting scalers on training data only...")
    
    train_inputs_list = []
    train_targets_list = []
    
    with h5py.File(raw_data_path, 'r') as f:
        for idx in train_indices:
            key = sample_keys[idx]
            group = f[key]
            
            # Load charge density (input)
            rho = group['rho'][:]  # shape: (Nx, Ny, Nz)
            train_inputs_list.append(rho)
            
            # Load electric field (target) and transpose
            efield = group['efield'][:]  # shape: (Nx, Ny, Nz, 3)
            efield = np.transpose(efield, (3, 0, 1, 2))  # shape: (3, Nx, Ny, Nz)
            train_targets_list.append(efield)
    
    # Convert to numpy arrays and reshape for scaler
    train_inputs = np.array(train_inputs_list)  # shape: (n_train, Nx, Ny, Nz)
    train_targets = np.array(train_targets_list)  # shape: (n_train, 3, Nx, Ny, Nz)
    
    logger.info(f"Training data shapes: inputs={train_inputs.shape}, targets={train_targets.shape}")
    
    # Reshape for scaler fitting (2D arrays required)
    input_flat = train_inputs.reshape(-1, 1)  # shape: (n_train*Nx*Ny*Nz, 1)
    target_flat = train_targets.reshape(-1, 1)  # shape: (n_train*3*Nx*Ny*Nz, 1)
    
    # Fit scalers
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    input_scaler.fit(input_flat)
    target_scaler.fit(target_flat)
    
    logger.info(f"Input scaler: mean={input_scaler.mean_[0]:.6f}, std={input_scaler.scale_[0]:.6f}")
    logger.info(f"Target scaler: mean={target_scaler.mean_[0]:.6f}, std={target_scaler.scale_[0]:.6f}")
    
    return input_scaler, target_scaler


def process_and_save_split(
    raw_data_path: str,
    output_path: str,
    indices: List[int],
    sample_keys: List[str],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    split_name: str,
    grid_shape: Tuple[int, int, int]
) -> None:
    """
    Process a data split and save to HDF5 file in monolithic format.
    
    Args:
        raw_data_path: Path to raw HDF5 file
        output_path: Path for output HDF5 file
        indices: Sample indices for this split
        sample_keys: List of all sample keys
        input_scaler: Fitted scaler for inputs
        target_scaler: Fitted scaler for targets
        split_name: Name of split (for logging)
    """
    logger.info(f"Processing {split_name} split ({len(indices)} samples)...")
    
    with h5py.File(raw_data_path, 'r') as input_file, \
         h5py.File(output_path, 'w') as output_file:
        
        # Create output datasets with correct shapes (using passed grid_shape)
        charge_density_shape = (len(indices), *grid_shape)
        electric_field_shape = (len(indices), 3, *grid_shape)  # Note: 3 first due to transpose
        
        charge_density_ds = output_file.create_dataset(
            'charge_density',
            shape=charge_density_shape,
            dtype=np.float32,  # Use float32 for PyTorch compatibility
            compression='gzip'
        )
        
        electric_field_ds = output_file.create_dataset(
            'electric_field', 
            shape=electric_field_shape,
            dtype=np.float32,
            compression='gzip'
        )
        
        # Process each sample
        for i, idx in enumerate(indices):
            key = sample_keys[idx]
            group = input_file[key]
            
            # Process charge density (input)
            rho = group['rho'][:]
            rho_flat = rho.reshape(-1, 1)
            rho_normalized = input_scaler.transform(rho_flat)
            rho_normalized = rho_normalized.reshape(rho.shape).astype(np.float32)
            charge_density_ds[i] = rho_normalized
            
            # Process electric field (target)
            efield = group['efield'][:]  # shape: (Nx, Ny, Nz, 3)
            efield = np.transpose(efield, (3, 0, 1, 2))  # shape: (3, Nx, Ny, Nz)
            efield_flat = efield.reshape(-1, 1)
            efield_normalized = target_scaler.transform(efield_flat)
            efield_normalized = efield_normalized.reshape(efield.shape).astype(np.float32)
            electric_field_ds[i] = efield_normalized
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(indices)} samples")
        
        # Add metadata
        output_file.attrs['split_name'] = split_name
        output_file.attrs['num_samples'] = len(indices)
        output_file.attrs['grid_shape'] = grid_shape
        output_file.attrs['normalized'] = True
        
    logger.info(f"Saved {split_name} data to {output_path}")


def save_scalers(input_scaler: StandardScaler, target_scaler: StandardScaler, save_path: str) -> None:
    """Save fitted scalers to disk."""
    scalers = {
        'input_scaler': input_scaler,
        'target_scaler': target_scaler
    }
    joblib.dump(scalers, save_path)
    logger.info(f"Saved scalers to {save_path}")


def main(config_path: str = "configs/training_config.yaml") -> None:
    """Main preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline...")
    
    # Load configuration
    config = load_config(config_path)
    paths = config['paths']
    preprocessing = config['preprocessing']
    
    # Create output directories
    os.makedirs(paths['processed_dir'], exist_ok=True)
    os.makedirs(paths['model_save_dir'], exist_ok=True)
    
    # Examine raw data
    sample_keys, grid_shape = examine_raw_data(paths['raw_data_path'])
    n_samples = len(sample_keys)
    
    # Split data
    train_indices, val_indices, test_indices = split_indices(
        n_samples, 
        preprocessing['split_ratios'],
        preprocessing['split_seed']
    )
    
    # Fit scalers on training data ONLY
    input_scaler, target_scaler = fit_scalers(
        paths['raw_data_path'],
        train_indices,
        sample_keys
    )
    
    # Save scalers
    scaler_path = os.path.join(paths['model_save_dir'], 'scalers.pkl')
    save_scalers(input_scaler, target_scaler, scaler_path)
    
    # Process and save each split
    splits = [
        ('train', train_indices),
        ('val', val_indices), 
        ('test', test_indices)
    ]
    
    for split_name, indices in splits:
        output_path = os.path.join(paths['processed_dir'], f'{split_name}.h5')
        process_and_save_split(
            paths['raw_data_path'],
            output_path,
            indices,
            sample_keys,
            input_scaler,
            target_scaler,
            split_name,
            grid_shape
        )
    
    logger.info("Data preprocessing completed successfully!")
    
    # Verify normalization
    logger.info("Verifying normalization...")
    with h5py.File(os.path.join(paths['processed_dir'], 'train.h5'), 'r') as f:
        charge_data = f['charge_density'][:]
        field_data = f['electric_field'][:]
        
        logger.info(f"Charge density: mean={charge_data.mean():.6f}, std={charge_data.std():.6f}")
        logger.info(f"Electric field: mean={field_data.mean():.6f}, std={field_data.std():.6f}")


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml"
    main(config_path) 