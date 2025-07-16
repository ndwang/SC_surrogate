"""
Comprehensive test suite for Space Charge Surrogate Model training pipeline.

Tests model instantiation, forward pass, training loop functionality,
checkpoint saving/loading, and evaluation pipeline.
"""

import pytest
import torch
import numpy as np
import os
import yaml
import tempfile
import shutil
from pathlib import Path
import h5py
import pickle
import logging
import gc
import time

# Import our modules
import sys
sys.path.append('.')
from modeling.models import get_model, create_model_from_config, list_available_models
from modeling.models.cnn3d import CNN3D
from modeling.train import Trainer
from evaluation.evaluate import Evaluator


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
def setup_test_environment():
    """
    Create a complete test environment with config, fake data, and all necessary files.
    """
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(test_dir, "data", "raw")
    processed_dir = os.path.join(test_dir, "data", "processed")
    model_dir = os.path.join(test_dir, "saved_models")
    config_dir = os.path.join(test_dir, "configs")
    log_dir = os.path.join(test_dir, "logs")
    
    for directory in [raw_dir, processed_dir, model_dir, config_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Create fake raw data
    raw_path = os.path.join(raw_dir, "test_data.h5")
    n_samples = 20  # Small for fast testing
    grid_size = (8, 8, 8)
    
    with h5py.File(raw_path, 'w') as f:
        for i in range(n_samples):
            group = f.create_group(f'run_{i:05d}')
            
            # Create realistic fake data
            rho = np.random.exponential(0.001, grid_size).astype(np.float64)
            group.create_dataset('rho', data=rho)
            
            efield = np.random.normal(0, 1e6, (*grid_size, 3)).astype(np.float64)
            group.create_dataset('efield', data=efield)
            
            params = np.random.uniform(0.002, 0.007, 3).astype(np.float64)
            group.create_dataset('parameters', data=params)
    
    # Create test configuration
    config_path = os.path.join(config_dir, "test_config.yaml")
    config = {
        'paths': {
            'raw_data_path': raw_path,
            'processed_dir': processed_dir + "/",
            'model_save_dir': model_dir + "/",
            'log_dir': log_dir + "/"
        },
        'preprocessing': {
            'split_ratios': [0.8, 0.1, 0.1],
            'split_seed': 42,
            'normalization_method': 'standard',
            'shuffle_data': True,
            'use_chunked_processing': False,
            'chunk_size': 1000
        },
        'model': {
            'architecture': 'cnn3d',
            'input_channels': 1,
            'output_channels': 3,
            'hidden_channels': [16, 32, 16],  # Smaller for testing
            'kernel_size': 3,
            'padding': 1,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.1,
            'weight_init': 'kaiming_normal'
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 3,  # Very few epochs for testing
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 1e-5,
            'scheduler': {
                'type': 'plateau',
                'patience': 2,
                'factor': 0.5,
                'min_lr': 1e-6
            },
            'loss_function': 'mse',
            'validation_frequency': 1,
            'save_frequency': 1,
            'early_stopping': {
                'patience': 5,
                'min_delta': 1e-6
            },
            'device': 'cpu',  # Force CPU for consistent testing
            'num_workers': 0,   # No multiprocessing in tests
            'pin_memory': False
        },
        'evaluation': {
            'metrics': ['mse', 'mae', 'r2_score'],
            'save_predictions': True,
            'plot_frequency': 1,
            'max_plots': 2
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': True,
            'log_filename': 'test_training.log'
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
        'grid_size': grid_size,
        'config': config
    }
    
    yield test_info
    
    # Robust cleanup that handles Windows file locking
    robust_cleanup(test_dir)


def test_model_registry():
    """Test that the model registry works correctly."""
    # Check available models
    available_models = list_available_models()
    assert 'cnn3d' in available_models, "CNN3D model not registered"
    
    # Test model creation via registry
    test_config = {
        'model': {
            'architecture': 'cnn3d',
            'input_channels': 1,
            'output_channels': 3,
            'hidden_channels': [8, 16, 8],
            'kernel_size': 3,
            'padding': 1,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.1,
            'weight_init': 'kaiming_normal'
        }
    }
    
    # Test both ways to create model
    model1 = get_model('cnn3d', test_config)
    model2 = create_model_from_config(test_config)
    
    assert isinstance(model1, CNN3D), "Registry didn't return CNN3D instance"
    assert isinstance(model2, CNN3D), "Config creation didn't return CNN3D instance"
    
    # Test invalid model name
    with pytest.raises(ValueError):
        get_model('nonexistent_model', test_config)


def test_cnn3d_model():
    """Test CNN3D model instantiation and forward pass."""
    test_config = {
        'model': {
            'input_channels': 1,
            'output_channels': 3,
            'hidden_channels': [8, 16, 8],
            'kernel_size': 3,
            'padding': 1,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.1,
            'weight_init': 'kaiming_normal'
        }
    }
    
    # Create model
    model = CNN3D(test_config)
    
    # Check model properties
    summary = model.get_model_summary()
    assert summary['model_name'] == 'CNN3D'
    assert summary['input_channels'] == 1
    assert summary['output_channels'] == 3
    assert summary['total_parameters'] > 0
    
    # Test forward pass with different input sizes
    for grid_size in [(8, 8, 8), (16, 16, 16)]:
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, *grid_size)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 3, *grid_size)
        assert output.shape == expected_shape, f"Wrong output shape: {output.shape} vs {expected_shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Model output contains non-finite values"
    
    # Test training mode
    model.train()
    input_tensor = torch.randn(1, 1, 8, 8, 8)
    output = model(input_tensor)
    assert output.requires_grad, "Model output should require gradients in training mode"


def test_training_pipeline(setup_test_environment):
    """Test the complete training pipeline."""
    test_info = setup_test_environment
    config_path = test_info['config_path']
    model_dir = test_info['model_dir']
    
    # Initialize trainer
    trainer = Trainer(config_path)
    try:
        assert trainer.config is not None, "Config not loaded"
        assert trainer.device is not None, "Device not set"
        
        # Run preprocessing if needed
        trainer._ensure_data_processed()
        
        # Check that processed files exist
        processed_dir = Path(test_info['processed_dir'])
        assert (processed_dir / 'train.h5').exists(), "Training data not created"
        assert (processed_dir / 'val.h5').exists(), "Validation data not created"
        assert (processed_dir / 'test.h5').exists(), "Test data not created"
        
        # Setup data loaders
        trainer.setup_data()
        assert trainer.train_loader is not None, "Train loader not created"
        assert trainer.val_loader is not None, "Validation loader not created"
        assert trainer.test_loader is not None, "Test loader not created"
        
        # Setup model
        trainer.setup_model()
        assert trainer.model is not None, "Model not created"
        
        # Setup optimizer
        trainer.setup_optimizer()
        assert trainer.optimizer is not None, "Optimizer not created"
        
        # Test loss function setup
        criterion = trainer.setup_loss_function()
        assert isinstance(criterion, torch.nn.Module), "Loss function not created properly"
        
        # Run a few training steps (not full training to save time)
        trainer.current_epoch = 0
        
        # Test single epoch training
        initial_loss = trainer.train_epoch(criterion)
        assert isinstance(initial_loss, float), "Training didn't return loss value"
        assert initial_loss > 0, "Loss should be positive"
        
        # Test validation
        val_loss = trainer.validate(criterion)
        assert isinstance(val_loss, float), "Validation didn't return loss value"
        assert val_loss > 0, "Validation loss should be positive"
        
        # Test checkpoint saving
        trainer.save_checkpoint(0, val_loss, is_best=True)
        
        # Check that checkpoint files were created
        model_path = Path(model_dir)
        assert (model_path / 'best_model.pth').exists(), "Best model not saved"
        assert (model_path / 'latest_checkpoint.pth').exists(), "Latest checkpoint not saved"
        assert (model_path / 'checkpoint_epoch_000.pth').exists(), "Epoch checkpoint not saved"
    finally:
        # Ensure cleanup
        trainer.cleanup()


def test_checkpoint_loading(setup_test_environment):
    """Test checkpoint saving and loading functionality."""
    test_info = setup_test_environment
    config_path = test_info['config_path']
    model_dir = test_info['model_dir']
    
    # Create and train model briefly
    trainer = Trainer(config_path)
    try:
        trainer.setup_data()
        trainer.setup_model()
        trainer.setup_optimizer()
        criterion = trainer.setup_loss_function()
        
        # Get initial weights (create a deep copy)
        initial_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        
        # Ensure model is in training mode and run multiple steps to ensure weight changes
        trainer.model.train()
        trainer.current_epoch = 0
        
        # Run training for multiple epochs to ensure weights change
        for _ in range(3):
            trainer.train_epoch(criterion)
        
        val_loss = trainer.validate(criterion)
        trainer.save_checkpoint(0, val_loss, is_best=True)
        
        # Get trained weights
        trained_weights = trainer.model.state_dict()
        
        # Verify weights changed (with tolerance for floating point differences)
        weights_changed = False
        for key in initial_weights:
            if not torch.allclose(initial_weights[key], trained_weights[key], atol=1e-6):
                weights_changed = True
                break
        assert weights_changed, f"Model weights should have changed during training. Checked {len(initial_weights)} parameter groups."
    finally:
        trainer.cleanup()
    
    # Create new model and load checkpoint
    new_trainer = Trainer(config_path)
    try:
        new_trainer.setup_data()
        new_trainer.setup_model()
        
        # Load checkpoint
        checkpoint_path = Path(model_dir) / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path)
        new_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Compare weights
        loaded_weights = new_trainer.model.state_dict()
        
        for key in trained_weights:
            assert torch.equal(loaded_weights[key], trained_weights[key]), \
                f"Loaded weights differ from saved weights for {key}"
    finally:
        new_trainer.cleanup()


def test_evaluation_pipeline(setup_test_environment):
    """Test the evaluation pipeline."""
    test_info = setup_test_environment
    config_path = test_info['config_path']
    
    # First train a model to have something to evaluate
    trainer = Trainer(config_path)
    try:
        trainer.setup_data()
        trainer.setup_model()
        trainer.setup_optimizer()
        criterion = trainer.setup_loss_function()
        
        # Train for minimal steps
        trainer.current_epoch = 0
        trainer.train_epoch(criterion)
        val_loss = trainer.validate(criterion)
        trainer.save_checkpoint(0, val_loss, is_best=True)
    finally:
        trainer.cleanup()
    
    # Now test evaluation
    evaluator = Evaluator(config_path)
    try:
        # Test checkpoint finding
        assert evaluator.checkpoint_path is not None, "Checkpoint path not found"
        assert os.path.exists(evaluator.checkpoint_path), "Checkpoint file doesn't exist"
        
        # Load model
        evaluator.load_model()
        assert evaluator.model is not None, "Model not loaded"
        
        # Setup data
        evaluator.setup_data()
        assert evaluator.test_loader is not None, "Test loader not created"
        
        # Run inference
        predictions, targets = evaluator.predict_all()
        assert predictions.shape[0] > 0, "No predictions generated"
        assert predictions.shape == targets.shape, "Prediction and target shapes don't match"
        assert predictions.shape[1] == 3, "Wrong number of output channels"
        
        # Compute metrics
        metrics = evaluator.compute_metrics(predictions, targets)
        
        # Check that all expected metrics are present
        assert 'mse' in metrics, "MSE metric not computed"
        assert 'mae' in metrics, "MAE metric not computed"
        assert 'r2_score' in metrics, "R² metric not computed"
        assert 'rmse' in metrics, "RMSE metric not computed"
        assert 'per_component' in metrics, "Per-component metrics not computed"
        
        # Check per-component metrics
        per_comp = metrics['per_component']
        assert 'Ex' in per_comp, "Ex component metrics not computed"
        assert 'Ey' in per_comp, "Ey component metrics not computed"
        assert 'Ez' in per_comp, "Ez component metrics not computed"
        
        # Check that metrics are reasonable (not NaN or negative for metrics that shouldn't be)
        assert not np.isnan(metrics['mse']), "MSE is NaN"
        assert not np.isnan(metrics['mae']), "MAE is NaN"
        assert metrics['mse'] >= 0, "MSE should be non-negative"
        assert metrics['mae'] >= 0, "MAE should be non-negative"
    finally:
        evaluator.cleanup()


def test_end_to_end_training(setup_test_environment):
    """Test complete end-to-end training process."""
    test_info = setup_test_environment
    config_path = test_info['config_path']
    model_dir = test_info['model_dir']
    
    # Run complete training
    trainer = Trainer(config_path)
    try:
        trainer.train()  # This runs the full pipeline
        
        # Check that training completed successfully
        assert len(trainer.train_losses) > 0, "No training losses recorded"
        assert len(trainer.val_losses) > 0, "No validation losses recorded"
        
        # Check that model files were saved
        model_path = Path(model_dir)
        assert (model_path / 'best_model.pth').exists(), "Best model not saved"
        assert (model_path / 'training_history.pkl').exists(), "Training history not saved"
        
        # Check training history
        with open(model_path / 'training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        assert 'train_losses' in history, "Training losses not in history"
        assert 'val_losses' in history, "Validation losses not in history"
        assert 'best_val_loss' in history, "Best validation loss not in history"
        assert len(history['train_losses']) == len(trainer.train_losses), "History length mismatch"
    finally:
        trainer.cleanup()


def test_model_reproducibility(setup_test_environment):
    """Test that model training is reproducible with same config."""
    test_info = setup_test_environment
    config_path = test_info['config_path']
    
    # Train two models with same config and seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    trainer1 = Trainer(config_path)
    try:
        trainer1.setup_data()
        trainer1.setup_model()
        trainer1.setup_optimizer()
        
        # Get initial weights
        weights1 = trainer1.model.state_dict()
    finally:
        trainer1.cleanup()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    trainer2 = Trainer(config_path)
    try:
        trainer2.setup_data()
        trainer2.setup_model()
        trainer2.setup_optimizer()
        
        # Get initial weights
        weights2 = trainer2.model.state_dict()
        
        # Compare initial weights (should be identical with same seed)
        for key in weights1:
            assert torch.equal(weights1[key], weights2[key]), \
                f"Initial weights differ for {key} (reproducibility issue)"
    finally:
        trainer2.cleanup()


def test_error_handling():
    """Test error handling for various edge cases."""
    # Test invalid model architecture
    invalid_config = {
        'model': {
            'architecture': 'nonexistent_model',
            'input_channels': 1,
            'output_channels': 3
        }
    }
    
    with pytest.raises(ValueError):
        create_model_from_config(invalid_config)
    
    # Test missing architecture in config
    missing_arch_config = {
        'model': {
            'input_channels': 1,
            'output_channels': 3
        }
    }
    
    with pytest.raises(ValueError):
        create_model_from_config(missing_arch_config)
    
    # Note: Invalid optimizer would be tested in the actual Trainer class
    # during initialization, but we focus on the model registry tests here


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 