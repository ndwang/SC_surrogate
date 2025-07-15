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


class Preprocessor:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.paths = self.config['paths']
        self.preprocessing = self.config['preprocessing']
        self.sample_keys = None
        self.grid_shape = None
        self.n_samples = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.input_scaler = None
        self.target_scaler = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config

    def examine_raw_data(self):
        """
        Examine raw HDF5 file to get sample keys and data dimensions.
        Sets self.sample_keys and self.grid_shape.
        """
        raw_data_path = self.paths['raw_data_path']
        logger.info(f"Examining raw data file: {raw_data_path}")
        with h5py.File(raw_data_path, 'r') as f:
            sample_keys = list(f.keys())
            first_key = sample_keys[0]
            rho_shape = f[first_key]['rho'].shape
            efield_shape = f[first_key]['efield'].shape
            logger.info(f"Found {len(sample_keys)} samples")
            logger.info(f"rho shape: {rho_shape}, efield shape: {efield_shape}")
            self.sample_keys = sample_keys
            self.grid_shape = rho_shape
            self.n_samples = len(sample_keys)

    def split_indices(self):
        """
        Split sample indices into train, validation, and test sets.
        Sets self.train_indices, self.val_indices, self.test_indices.
        """
        n_samples = self.n_samples
        split_ratios = self.preprocessing['split_ratios']
        seed = self.preprocessing['split_seed']
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(split_ratios)}"
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(n_samples)
        train_size = int(n_samples * split_ratios[0])
        val_size = int(n_samples * split_ratios[1])
        train_indices = shuffled_indices[:train_size].tolist()
        val_indices = shuffled_indices[train_size:train_size + val_size].tolist()
        test_indices = shuffled_indices[train_size + val_size:].tolist()
        logger.info(f"Data split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def fit_scalers(self):
        """
        Fit StandardScaler objects on training data ONLY.
        Sets self.input_scaler and self.target_scaler.
        """
        raw_data_path = self.paths['raw_data_path']
        train_indices = self.train_indices
        sample_keys = self.sample_keys
        logger.info("Fitting scalers on training data...")
        train_inputs_list = []
        train_targets_list = []
        with h5py.File(raw_data_path, 'r') as f:
            for idx in train_indices:
                key = sample_keys[idx]
                group = f[key]
                rho = group['rho'][:]
                train_inputs_list.append(rho)
                efield = group['efield'][:]
                efield = np.transpose(efield, (3, 0, 1, 2))
                train_targets_list.append(efield)
        train_inputs = np.array(train_inputs_list)
        train_targets = np.array(train_targets_list)
        logger.info(f"Training data shapes: inputs={train_inputs.shape}, targets={train_targets.shape}")
        input_flat = train_inputs.reshape(-1, 1)
        target_flat = train_targets.reshape(-1, 1)
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(input_flat)
        target_scaler.fit(target_flat)
        logger.info(f"Input scaler: mean={input_scaler.mean_[0]:.6f}, std={input_scaler.scale_[0]:.6f}")
        logger.info(f"Target scaler: mean={target_scaler.mean_[0]:.6f}, std={target_scaler.scale_[0]:.6f}")
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def process_and_save_split(self, indices: List[int], split_name: str):
        """
        Process a data split and save to HDF5 file in monolithic format.
        """
        raw_data_path = self.paths['raw_data_path']
        output_path = os.path.join(self.paths['processed_dir'], f'{split_name}.h5')
        sample_keys = self.sample_keys
        input_scaler = self.input_scaler
        target_scaler = self.target_scaler
        grid_shape = self.grid_shape
        logger.info(f"Processing {split_name} split ({len(indices)} samples)...")
        with h5py.File(raw_data_path, 'r') as input_file, \
             h5py.File(output_path, 'w') as output_file:
            charge_density_shape = (len(indices), *grid_shape)
            electric_field_shape = (len(indices), 3, *grid_shape)
            charge_density_ds = output_file.create_dataset(
                'charge_density',
                shape=charge_density_shape,
                dtype=np.float32,
                compression='gzip'
            )
            electric_field_ds = output_file.create_dataset(
                'electric_field',
                shape=electric_field_shape,
                dtype=np.float32,
                compression='gzip'
            )
            for i, idx in enumerate(indices):
                key = sample_keys[idx]
                group = input_file[key]
                rho = group['rho'][:]
                rho_flat = rho.reshape(-1, 1)
                rho_normalized = input_scaler.transform(rho_flat)
                rho_normalized = rho_normalized.reshape(rho.shape).astype(np.float32)
                charge_density_ds[i] = rho_normalized
                efield = group['efield'][:]
                efield = np.transpose(efield, (3, 0, 1, 2))
                efield_flat = efield.reshape(-1, 1)
                efield_normalized = target_scaler.transform(efield_flat)
                efield_normalized = efield_normalized.reshape(efield.shape).astype(np.float32)
                electric_field_ds[i] = efield_normalized
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(indices)} samples")
            output_file.attrs['split_name'] = split_name
            output_file.attrs['num_samples'] = len(indices)
            output_file.attrs['grid_shape'] = grid_shape
            output_file.attrs['normalized'] = True
        logger.info(f"Saved {split_name} data to {output_path}")

    def save_scalers(self):
        """Save fitted scalers to disk."""
        scaler_path = os.path.join(self.paths['model_save_dir'], 'scalers.pkl')
        scalers = {
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler
        }
        joblib.dump(scalers, scaler_path)
        logger.info(f"Saved scalers to {scaler_path}")

    def verify_normalization(self):
        """Verify normalization on the training set."""
        train_path = os.path.join(self.paths['processed_dir'], 'train.h5')
        logger.info("Verifying normalization...")
        with h5py.File(train_path, 'r') as f:
            charge_data = f['charge_density'][:]
            field_data = f['electric_field'][:]
            logger.info(f"Charge density: mean={charge_data.mean():.6f}, std={charge_data.std():.6f}")
            logger.info(f"Electric field: mean={field_data.mean():.6f}, std={field_data.std():.6f}")

    def run(self):
        logger.info("Starting data preprocessing pipeline...")
        os.makedirs(self.paths['processed_dir'], exist_ok=True)
        os.makedirs(self.paths['model_save_dir'], exist_ok=True)
        self.examine_raw_data()
        self.split_indices()
        self.fit_scalers()
        self.save_scalers()
        splits = [
            ('train', self.train_indices),
            ('val', self.val_indices),
            ('test', self.test_indices)
        ]
        for split_name, indices in splits:
            self.process_and_save_split(indices, split_name)
        logger.info("Data preprocessing completed successfully!")
        self.verify_normalization()


def main(config_path: str = "configs/training_config.yaml") -> None:
    preprocessor = Preprocessor(config_path)
    preprocessor.run()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml"
    main(config_path) 