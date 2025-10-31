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
from typing import Dict, List, Any
from preprocessing.scalers import get_scaler, get_fitted_attributes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config_path: str = "configs/training_config.yaml", batch_size: int = 32):
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
        self.batch_size = batch_size

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
        Fit scaler objects on training data using config-specified types.
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
        # Require scaler configs to be dicts (matching new config format)
        input_scaler_cfg = self.preprocessing.get('input_scaler')
        target_scaler_cfg = self.preprocessing.get('target_scaler')
        if not isinstance(input_scaler_cfg, dict):
            raise ValueError("input_scaler config must be a dictionary with at least a 'type' key (see training_config.yaml)")
        if not isinstance(target_scaler_cfg, dict):
            raise ValueError("target_scaler config must be a dictionary with at least a 'type' key (see training_config.yaml)")
        def parse_scaler(cfg):
            scaler_type = cfg.get('type', 'standard')
            kwargs = {k: v for k, v in cfg.items() if k != 'type'}
            return scaler_type, kwargs
        input_scaler_type, input_scaler_kwargs = parse_scaler(input_scaler_cfg)
        target_scaler_type, target_scaler_kwargs = parse_scaler(target_scaler_cfg)
        input_scaler = get_scaler(input_scaler_type, **input_scaler_kwargs)
        target_scaler = get_scaler(target_scaler_type, **target_scaler_kwargs)
        input_scaler.fit(input_flat)
        target_scaler.fit(target_flat)
        logger.info(f"Input scaler ({input_scaler_type}): fitted. Params: {get_fitted_attributes(input_scaler)}")
        logger.info(f"Target scaler ({target_scaler_type}): fitted. Params: {get_fitted_attributes(target_scaler)}")
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def process_and_save_split(self, indices: List[int], split_name: str, batch_size: int = 32):
        """
        Process a data split and save to HDF5 file in monolithic format, using batching for speed.
        """
        raw_data_path = self.paths['raw_data_path']
        output_path = os.path.join(self.paths['processed_dir'], f'{split_name}.h5')
        sample_keys = self.sample_keys
        input_scaler = self.input_scaler
        target_scaler = self.target_scaler
        grid_shape = self.grid_shape
        logger.info(f"Processing {split_name} split ({len(indices)} samples) with batch_size={batch_size}...")
        with h5py.File(raw_data_path, 'r') as input_file, \
             h5py.File(output_path, 'w') as output_file:
            charge_density_shape = (len(indices), *grid_shape)
            electric_field_shape = (len(indices), 3, *grid_shape)
            # Use sample-aligned chunking and fast LZF compression for faster reads
            charge_density_ds = output_file.create_dataset(
                'charge_density',
                shape=charge_density_shape,
                dtype=np.float32,
                chunks=(1, *grid_shape),
                compression='lzf'
            )
            electric_field_ds = output_file.create_dataset(
                'electric_field',
                shape=electric_field_shape,
                dtype=np.float32,
                chunks=(1, 3, *grid_shape),
                compression='lzf'
            )
            n = len(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_indices = indices[start:end]
                # Read batch
                batch_rho = []
                batch_efield = []
                for idx in batch_indices:
                    key = sample_keys[idx]
                    group = input_file[key]
                    rho = group['rho'][:]
                    batch_rho.append(rho)
                    efield = group['efield'][:]
                    efield = np.transpose(efield, (3, 0, 1, 2))
                    batch_efield.append(efield)
                batch_rho = np.array(batch_rho)
                batch_efield = np.array(batch_efield)
                # Normalize batch
                batch_rho_flat = batch_rho.reshape(-1, 1)
                batch_rho_norm = input_scaler.transform(batch_rho_flat)
                batch_rho_norm = batch_rho_norm.reshape(batch_rho.shape).astype(np.float32)
                batch_efield_flat = batch_efield.reshape(-1, 1)
                batch_efield_norm = target_scaler.transform(batch_efield_flat)
                batch_efield_norm = batch_efield_norm.reshape(batch_efield.shape).astype(np.float32)
                # Write batch
                charge_density_ds[start:end] = batch_rho_norm
                electric_field_ds[start:end] = batch_efield_norm
                logger.info(f"Processed {end}/{n} samples")
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
            self.process_and_save_split(indices, split_name, batch_size=self.batch_size)
        logger.info("Data preprocessing completed successfully!")
        self.verify_normalization() 