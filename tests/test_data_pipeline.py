"""
Comprehensive test suite for Space Charge Surrogate Model data pipeline.

Tests the complete pipeline from raw data preprocessing to PyTorch Dataset
functionality, including proper data format conversion, normalization,
and tensor shape handling.
"""

import pytest
import torch
import h5py
import numpy as np
import os
import yaml
import tempfile
import shutil
import joblib
import logging
import gc
import time

# Import our modules
import sys
sys.path.append('.')
from preprocessing.preprocess_data import Preprocessor
from modeling.dataset import SpaceChargeDataset, create_data_loaders


def robust_cleanup(test_dir: str, max_attempts: int = 3, delay: float = 0.5):
    """
    Robust cleanup function that handles Windows file locking issues.
    
    This function:
    1. Closes all logging handlers
    2. Forces garbage collection
    3. Retries deletion with exponential backoff
    4. Handles individual file removal if directory removal fails
    """
    # Close all logging handlers to release file locks
    loggers = [logging.getLogger(name) for name in logging.getLogger().manager.loggerDict]
    loggers.append(logging.getLogger())  # Add root logger
    
    for logger in loggers:
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Force garbage collection to release file handles
    gc.collect()
    
    # Attempt to remove directory with retries
    for attempt in range(max_attempts):
        try:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            return  # Success
        except (OSError, PermissionError):
            if attempt < max_attempts - 1:
                # Wait and retry
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                # Last attempt - try to remove individual files
                try:
                    for root, dirs, files in os.walk(test_dir, topdown=False):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except Exception:
                                pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except Exception:
                                pass
                    os.rmdir(test_dir)
                except Exception:
                    # If all else fails, just warn and continue
                    print(f"Warning: Could not fully clean up test directory {test_dir}")


@pytest.fixture(scope="module")
def setup_pipeline_test_data():
    """
    Creates fake raw data file with the group-per-sample structure
    and configuration file for testing the complete pipeline.
    """
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(test_dir, "data", "raw")
    processed_dir = os.path.join(test_dir, "data", "processed")
    model_dir = os.path.join(test_dir, "saved_models")
    config_dir = os.path.join(test_dir, "configs")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Create fake raw data with group-per-sample structure
    raw_path = os.path.join(raw_dir, "test_space_charge_data.h5")
    n_samples = 50  # Small number for fast testing
    grid_size = (16, 16, 16)  # Smaller grid for faster testing
    
    with h5py.File(raw_path, 'w') as f:
        for i in range(n_samples):
            group = f.create_group(f'run_{i:05d}')
            
            # Create realistic fake data
            # Charge density: positive values, small magnitude
            rho = np.random.exponential(0.001, grid_size).astype(np.float64)
            group.create_dataset('rho', data=rho)
            
            # Electric field: can be negative/positive, larger magnitude  
            # Shape: (Nx, Ny, Nz, 3) as discovered in real data
            efield = np.random.normal(0, 1e6, (*grid_size, 3)).astype(np.float64)
            group.create_dataset('efield', data=efield)
            
            # Parameters: LHS sampling parameters
            params = np.random.uniform(0.002, 0.007, 3).astype(np.float64)
            group.create_dataset('parameters', data=params)
    
    # Create test configuration file
    config_path = os.path.join(config_dir, "test_training_config.yaml")
    config = {
        'paths': {
            'raw_data_path': raw_path,
            'processed_dir': processed_dir + "/",
            'model_save_dir': model_dir + "/"
        },
        'preprocessing': {
            'split_ratios': [0.8, 0.1, 0.1],
            'split_seed': 42,
            'normalization_method': 'standard',
            'shuffle_data': True,
            'use_chunked_processing': False,
            'chunk_size': 1000
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Store paths for tests
    test_info = {
        'test_dir': test_dir,
        'config_path': config_path,
        'raw_path': raw_path,
        'processed_dir': processed_dir,
        'model_dir': model_dir,
        'n_samples': n_samples,
        'grid_size': grid_size
    }
    
    yield test_info
    
    # Robust cleanup that handles Windows file locking
    robust_cleanup(test_dir)


def test_preprocess_data_script(setup_pipeline_test_data):
    """
    Tests that the preprocessor correctly reads the grouped format 
    and writes a monolithic format with proper normalization.
    """
    test_info = setup_pipeline_test_data
    config_path = test_info['config_path']
    processed_dir = test_info['processed_dir']
    model_dir = test_info['model_dir']
    n_samples = test_info['n_samples']
    grid_size = test_info['grid_size']
    
    # Run preprocessing
    Preprocessor(config_path).run()
    
    # Assert files were created
    train_path = os.path.join(processed_dir, "train.h5")
    val_path = os.path.join(processed_dir, "val.h5")
    test_path = os.path.join(processed_dir, "test.h5")
    scaler_path = os.path.join(processed_dir, "scalers.pkl")
    
    assert os.path.exists(train_path), "train.h5 was not created"
    assert os.path.exists(val_path), "val.h5 was not created"
    assert os.path.exists(test_path), "test.h5 was not created"
    assert os.path.exists(scaler_path), "scalers.pkl was not created"
    
    # Test the OUTPUT format is monolithic (not group-per-sample)
    with h5py.File(train_path, 'r') as f:
        assert 'charge_density' in f, "charge_density dataset not found"
        assert 'electric_field' in f, "electric_field dataset not found"
        
        # Check shapes - should be monolithic
        expected_train_samples = int(n_samples * 0.8)  # 80% for training
        charge_shape = f['charge_density'].shape
        field_shape = f['electric_field'].shape
        
        assert charge_shape == (expected_train_samples, *grid_size), \
            f"Wrong charge density shape: {charge_shape}"
        assert field_shape == (expected_train_samples, 3, *grid_size), \
            f"Wrong electric field shape: {field_shape}"
        
        # Check data types (should be float32 for PyTorch)
        assert f['charge_density'].dtype == np.float32, \
            f"Wrong dtype for charge_density: {f['charge_density'].dtype}"
        assert f['electric_field'].dtype == np.float32, \
            f"Wrong dtype for electric_field: {f['electric_field'].dtype}"
        
        # Test normalization - data should be approximately normalized
        charge_data = f['charge_density'][:]
        field_data = f['electric_field'][:]
        
        # Allow some tolerance due to finite sample size
        assert abs(charge_data.mean()) < 0.1, \
            f"Charge data not normalized: mean={charge_data.mean()}"
        assert abs(charge_data.std() - 1.0) < 0.2, \
            f"Charge data not normalized: std={charge_data.std()}"
        
        assert abs(field_data.mean()) < 0.1, \
            f"Field data not normalized: mean={field_data.mean()}"
        assert abs(field_data.std() - 1.0) < 0.2, \
            f"Field data not normalized: std={field_data.std()}"
        
        # Check metadata
        assert f.attrs['normalized'], "Normalized flag not set"
        assert f.attrs['num_samples'] == expected_train_samples
    
    # Test validation and test sets
    with h5py.File(val_path, 'r') as f:
        expected_val_samples = int(n_samples * 0.1)
        assert f['charge_density'].shape[0] == expected_val_samples
        assert f['electric_field'].shape[0] == expected_val_samples
    
    with h5py.File(test_path, 'r') as f:
        expected_test_samples = n_samples - expected_train_samples - expected_val_samples
        assert f['charge_density'].shape[0] == expected_test_samples
        assert f['electric_field'].shape[0] == expected_test_samples
    
    # Test scalers can be loaded
    scalers = joblib.load(scaler_path)
    assert 'input_scaler' in scalers, "input_scaler not found in saved scalers"
    assert 'target_scaler' in scalers, "target_scaler not found in saved scalers"
    
    # Test scalers have expected properties
    input_scaler = scalers['input_scaler']
    target_scaler = scalers['target_scaler']
    
    assert hasattr(input_scaler, 'mean_'), "Input scaler not fitted"
    assert hasattr(input_scaler, 'scale_'), "Input scaler not fitted"
    assert hasattr(target_scaler, 'mean_'), "Target scaler not fitted"
    assert hasattr(target_scaler, 'scale_'), "Target scaler not fitted"


def test_space_charge_dataset(setup_pipeline_test_data):
    """Tests the Dataset class on the PROCESSED data file."""
    test_info = setup_pipeline_test_data
    processed_dir = test_info['processed_dir']
    grid_size = test_info['grid_size']
    
    # First run preprocessing to create the processed files
    Preprocessor(test_info['config_path']).run()
    
    # Test train dataset
    train_path = os.path.join(processed_dir, "train.h5")
    train_dataset = SpaceChargeDataset(h5_path=train_path)
    
    expected_train_samples = int(test_info['n_samples'] * 0.8)
    assert len(train_dataset) == expected_train_samples, \
        f"Wrong dataset length: {len(train_dataset)}"
    
    # Test single sample loading
    input_tensor, target_tensor = train_dataset[0]
    
    # Check tensor shapes
    assert input_tensor.shape == (1, *grid_size), \
        f"Wrong input shape: {input_tensor.shape}"
    assert target_tensor.shape == (3, *grid_size), \
        f"Wrong target shape: {target_tensor.shape}"
    
    # Check tensor dtypes
    assert input_tensor.dtype == torch.float32, \
        f"Wrong input dtype: {input_tensor.dtype}"
    assert target_tensor.dtype == torch.float32, \
        f"Wrong target dtype: {target_tensor.dtype}"
    
    # Test DataLoader integration
    batch_size = 4
    loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # No multiprocessing for tests
    )
    
    batch_input, batch_target = next(iter(loader))
    
    # Check batch shapes
    assert batch_input.shape == (batch_size, 1, *grid_size), \
        f"Wrong batch input shape: {batch_input.shape}"
    assert batch_target.shape == (batch_size, 3, *grid_size), \
        f"Wrong batch target shape: {batch_target.shape}"
    
    # Test dataset utilities
    sample_info = train_dataset.get_sample_info(0)
    assert 'index' in sample_info
    assert 'grid_shape' in sample_info
    
    stats = train_dataset.get_data_statistics(max_samples=5)
    assert 'charge_density' in stats
    assert 'electric_field' in stats
    
    # Clean up
    train_dataset.close()
    del train_dataset


def test_data_loader_creation(setup_pipeline_test_data):
    """Test the create_data_loaders utility function."""
    test_info = setup_pipeline_test_data
    processed_dir = test_info['processed_dir']
    
    # Run preprocessing first
    Preprocessor(test_info['config_path']).run()
    
    # Test data loader creation
    train_path = os.path.join(processed_dir, "train.h5")
    val_path = os.path.join(processed_dir, "val.h5")
    test_path = os.path.join(processed_dir, "test.h5")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=2,
        num_workers=0,  # No multiprocessing for tests
        device='cpu'
    )
    
    # Test that loaders work
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    # All batches should have the same structure
    for batch in [train_batch, val_batch, test_batch]:
        input_tensor, target_tensor = batch
        assert input_tensor.shape[1] == 1, "Wrong input channels"
        assert target_tensor.shape[1] == 3, "Wrong target channels"
        assert input_tensor.dtype == torch.float32
        assert target_tensor.dtype == torch.float32


def test_error_handling(setup_pipeline_test_data):
    """Test error handling for edge cases."""
    test_info = setup_pipeline_test_data
    processed_dir = test_info['processed_dir']
    
    # Run preprocessing first
    Preprocessor(test_info['config_path']).run()
    
    train_path = os.path.join(processed_dir, "train.h5")
    dataset = SpaceChargeDataset(train_path)
    
    # Test index out of range
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
    
    with pytest.raises(IndexError):
        _ = dataset.get_sample_info(len(dataset))
    
    # Test invalid file path
    with pytest.raises(FileNotFoundError):
        _ = SpaceChargeDataset("nonexistent_file.h5")
    
    dataset.close()


def test_normalization_consistency(setup_pipeline_test_data):
    """Test that normalization is consistent across splits."""
    test_info = setup_pipeline_test_data
    processed_dir = test_info['processed_dir']
    
    # Run preprocessing
    Preprocessor(test_info['config_path']).run()
    
    # Load all datasets and check normalization is similar
    split_stats = {}
    
    for split_name in ['train', 'val', 'test']:
        path = os.path.join(processed_dir, f"{split_name}.h5")
        dataset = SpaceChargeDataset(path)
        
        stats = dataset.get_data_statistics(max_samples=None)  # Use all samples
        split_stats[split_name] = stats
        
        dataset.close()
        # Ensure dataset is properly cleaned up
        del dataset
    
    # Check that all splits have similar normalization
    # (allowing some variation due to different sample sets)
    for split in ['val', 'test']:
        charge_mean_diff = abs(
            split_stats['train']['charge_density']['mean'] - 
            split_stats[split]['charge_density']['mean']
        )
        field_mean_diff = abs(
            split_stats['train']['electric_field']['mean'] - 
            split_stats[split]['electric_field']['mean']
        )
        
        # Means should be close to 0 for all splits
        assert charge_mean_diff < 0.2, f"Charge mean too different in {split}: {charge_mean_diff}"
        assert field_mean_diff < 0.2, f"Field mean too different in {split}: {field_mean_diff}"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 