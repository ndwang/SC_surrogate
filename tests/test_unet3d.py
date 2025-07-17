"""
Comprehensive test suite for UNet3D model.

Tests model instantiation, forward pass, output shapes, parameter counts,
skip connections, and overfitting capability on synthetic data.
"""

import pytest
import torch
import numpy as np

# Import our modules
import sys
sys.path.append('.')
from modeling.models import get_model, create_model_from_config, list_available_models
from modeling.models.unet3d import UNet3D, EncoderBlock, DecoderBlock


class TestUNet3DComponents:
    """Test individual components of the UNet3D model."""
    
    def test_encoder_block(self):
        """Test EncoderBlock functionality."""
        in_channels, out_channels = 16, 32
        block = EncoderBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_batch_norm=True,
            dropout_rate=0.1,
            activation='relu'
        )
        
        # Test forward pass
        batch_size = 2
        spatial_size = (8, 8, 8)
        x = torch.randn(batch_size, in_channels, *spatial_size)
        
        pooled, skip = block(x)
        
        # Check output shapes
        expected_pooled_shape = (batch_size, out_channels, 4, 4, 4)  # Halved by MaxPool
        expected_skip_shape = (batch_size, out_channels, *spatial_size)  # Before pooling
        
        assert pooled.shape == expected_pooled_shape, f"Pooled shape {pooled.shape} != {expected_pooled_shape}"
        assert skip.shape == expected_skip_shape, f"Skip shape {skip.shape} != {expected_skip_shape}"
        
        # Check outputs are finite
        assert torch.isfinite(pooled).all(), "Pooled output contains non-finite values"
        assert torch.isfinite(skip).all(), "Skip output contains non-finite values"
    
    def test_decoder_block(self):
        """Test DecoderBlock functionality."""
        in_channels, skip_channels, out_channels = 64, 32, 32
        block = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_batch_norm=True,
            dropout_rate=0.1,
            activation='relu'
        )
        
        # Test forward pass
        batch_size = 2
        input_size = (4, 4, 4)  # Smaller input
        skip_size = (8, 8, 8)   # Larger skip (from encoder)
        
        x = torch.randn(batch_size, in_channels, *input_size)
        skip = torch.randn(batch_size, skip_channels, *skip_size)
        
        output = block(x, skip)
        
        # Check output shape should match skip spatial dimensions
        expected_shape = (batch_size, out_channels, *skip_size)
        assert output.shape == expected_shape, f"Output shape {output.shape} != {expected_shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Decoder output contains non-finite values"


class TestUNet3DModel:
    """Test the complete UNet3D model."""
    
    def test_unet3d_instantiation(self):
        """Test UNet3D model instantiation with various configurations."""
        # Test basic configuration
        config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 3,
                'initial_features': 16,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        model = UNet3D(config)
        summary = model.get_model_summary()
        
        # Check model properties
        assert summary['model_name'] == 'UNet3D'
        assert summary['input_channels'] == 1
        assert summary['output_channels'] == 3
        assert summary['depth'] == 3
        assert summary['initial_features'] == 16
        assert summary['total_parameters'] > 0
        assert summary['encoder_blocks'] == 3
        assert summary['decoder_blocks'] == 3
        
        # Test different depths
        for depth in [2, 3, 4]:
            config['model']['depth'] = depth
            model = UNet3D(config)
            summary = model.get_model_summary()
            assert summary['encoder_blocks'] == depth
            assert summary['decoder_blocks'] == depth
        
        # Test different features
        for features in [8, 16, 32]:
            config['model']['initial_features'] = features
            model = UNet3D(config)
            summary = model.get_model_summary()
            assert summary['initial_features'] == features
    
    def test_forward_pass_shapes(self):
        """Test forward pass with different input shapes."""
        config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 3,
                'initial_features': 16,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        model = UNet3D(config)
        model.eval()
        
        # Test different input sizes that work with depth=3 (need size divisible by 2^depth)
        test_sizes = [
            (8, 8, 8),    # 8 = 2^3, perfect fit
            (16, 16, 16), # 16 = 2^4, works
            (32, 32, 32), # 32 = 2^5, works
        ]
        
        for grid_size in test_sizes:
            batch_size = 2
            input_tensor = torch.randn(batch_size, 1, *grid_size)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Check output shape matches input spatial dimensions
            expected_shape = (batch_size, 3, *grid_size)
            assert output.shape == expected_shape, f"Grid {grid_size}: output {output.shape} != expected {expected_shape}"
            
            # Check output is finite
            assert torch.isfinite(output).all(), f"Model output contains non-finite values for grid {grid_size}"
    
    def test_skip_connections_gradient_flow(self):
        """Test that skip connections allow gradient flow."""
        config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,  # Smaller for faster testing
                'initial_features': 8,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.0,  # No dropout for gradient test
                'weight_init': 'kaiming_normal'
            }
        }
        
        model = UNet3D(config)
        model.train()
        
        # Create input that requires gradients
        input_tensor = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn(1, 3, 8, 8, 8)
        
        # Forward pass
        output = model(input_tensor)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that input gradients exist (skip connections should help)
        assert input_tensor.grad is not None, "No gradients flowing back to input"
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad)), "Zero gradients indicate potential vanishing gradient problem"
        
        # Check that model parameters have gradients
        param_grads_exist = any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) 
                               for p in model.parameters() if p.requires_grad)
        assert param_grads_exist, "No non-zero gradients in model parameters"
    
    def test_parameter_count_scaling(self):
        """Test that parameter count scales reasonably with model size."""
        base_config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,
                'initial_features': 8,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        # Test scaling with depth
        param_counts_depth = []
        for depth in [2, 3, 4]:
            config = base_config.copy()
            config['model']['depth'] = depth
            model = UNet3D(config)
            param_count = sum(p.numel() for p in model.parameters())
            param_counts_depth.append(param_count)
        
        # Parameters should increase with depth
        assert param_counts_depth[1] > param_counts_depth[0], "Parameters should increase with depth"
        assert param_counts_depth[2] > param_counts_depth[1], "Parameters should increase with depth"
        
        # Test scaling with features
        param_counts_features = []
        for features in [8, 16, 32]:
            config = base_config.copy()
            config['model']['initial_features'] = features
            model = UNet3D(config)
            param_count = sum(p.numel() for p in model.parameters())
            param_counts_features.append(param_count)
        
        # Parameters should increase significantly with features (quadratic relationship)
        assert param_counts_features[1] > param_counts_features[0] * 2, "Parameters should scale with features"
        assert param_counts_features[2] > param_counts_features[1] * 2, "Parameters should scale with features"
    
    def test_different_activations(self):
        """Test model with different activation functions."""
        base_config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,
                'initial_features': 8,
                'kernel_size': 3,
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        activations = ['relu', 'leaky_relu', 'elu', 'gelu']
        
        for activation in activations:
            config = base_config.copy()
            config['model']['activation'] = activation
            
            model = UNet3D(config)
            input_tensor = torch.randn(1, 1, 8, 8, 8)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Check output is finite for all activations
            assert torch.isfinite(output).all(), f"Non-finite output with {activation} activation"
            assert output.shape == (1, 3, 8, 8, 8), f"Wrong output shape with {activation} activation"


class TestUNet3DRegistration:
    """Test UNet3D integration with model registry."""
    
    def test_model_registry_contains_unet3d(self):
        """Test that UNet3D is registered in the model registry."""
        available_models = list_available_models()
        assert 'unet3d' in available_models, "UNet3D not found in model registry"
    
    def test_create_via_registry(self):
        """Test creating UNet3D via model registry."""
        config = {
            'model': {
                'architecture': 'unet3d',
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,
                'initial_features': 16,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        # Test both registry methods
        model1 = get_model('unet3d', config)
        model2 = create_model_from_config(config)
        
        assert isinstance(model1, UNet3D), "Registry didn't return UNet3D instance"
        assert isinstance(model2, UNet3D), "Config creation didn't return UNet3D instance"
        
        # Test forward pass
        input_tensor = torch.randn(1, 1, 8, 8, 8)
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        assert output1.shape == (1, 3, 8, 8, 8), "Wrong output shape from registry model"
        assert output2.shape == (1, 3, 8, 8, 8), "Wrong output shape from config model"


class TestUNet3DOverfitting:
    """Test UNet3D overfitting capability on synthetic data - crucial requirement from PRP."""
    
    def test_overfitting_synthetic_data(self):
        """Test that UNet3D can overfit a tiny synthetic dataset."""
        # Create a very simple synthetic dataset
        torch.manual_seed(42)  # For reproducibility
        np.random.seed(42)
        
        # Create synthetic data: simple pattern that model should learn
        batch_size = 4
        grid_size = (8, 8, 8)
        
        # Create input: random charge density
        inputs = torch.randn(batch_size, 1, *grid_size) * 0.1
        
        # Create target: simple linear transformation of input
        # This creates a deterministic relationship the model should learn
        targets = torch.zeros(batch_size, 3, *grid_size)
        targets[:, 0, :, :, :] = inputs[:, 0, :, :, :] * 2.0    # Ex = 2 * rho
        targets[:, 1, :, :, :] = inputs[:, 0, :, :, :] * -1.0   # Ey = -1 * rho
        targets[:, 2, :, :, :] = inputs[:, 0, :, :, :] * 0.5    # Ez = 0.5 * rho
        
        # Create small model for faster overfitting
        config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,  # Small depth
                'initial_features': 8,  # Small features
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': False,  # Disable batch norm for easier overfitting
                'dropout_rate': 0.0,  # Disable dropout for easier overfitting
                'weight_init': 'kaiming_normal'
            }
        }
        
        model = UNet3D(config)
        model.train()
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Train until overfitting
        initial_loss = None
        final_loss = None
        max_epochs = 200
        target_loss = 2e-4  # Very low loss indicates overfitting
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
            # Check if we've achieved overfitting
            if final_loss < target_loss:
                break
        
        # Verify overfitting occurred
        assert initial_loss is not None, "No initial loss recorded"
        assert final_loss is not None, "No final loss recorded"
        assert final_loss < initial_loss * 0.01, f"Model didn't overfit: initial={initial_loss:.6f}, final={final_loss:.6f}"
        assert final_loss < target_loss, f"Model didn't reach target loss: final={final_loss:.6f}, target={target_loss}"
        
        # Test that model produces consistent outputs (memorization)
        model.eval()
        with torch.no_grad():
            outputs1 = model(inputs)
            outputs2 = model(inputs)
        
        # Outputs should be identical (deterministic)
        assert torch.allclose(outputs1, outputs2), "Model outputs not deterministic"
        
        # Check that model learned the relationships (check correlations)
        # For Ex component: should correlate with 2 * input
        ex_pred = outputs1[:, 0, :, :, :].flatten()
        ex_target = targets[:, 0, :, :, :].flatten()
        ex_corr = torch.corrcoef(torch.stack([ex_pred, ex_target]))[0, 1]
        assert ex_corr > 0.8, f"Ex correlation too low: {ex_corr:.3f}"
        
        # For Ey component: should correlate with -1 * input  
        ey_pred = outputs1[:, 1, :, :, :].flatten()
        ey_target = targets[:, 1, :, :, :].flatten()
        ey_corr = torch.corrcoef(torch.stack([ey_pred, ey_target]))[0, 1]
        assert ey_corr > 0.8, f"Ey correlation too low: {ey_corr:.3f}"
        
        print(f"✓ Overfitting test passed: loss reduced from {initial_loss:.6f} to {final_loss:.6f} in {epoch+1} epochs")


class TestUNet3DEdgeCases:
    """Test UNet3D edge cases and error handling."""
    
    def test_minimal_configuration(self):
        """Test UNet3D with minimal configuration (using defaults)."""
        config = {'model': {}}  # Empty model config, should use defaults
        
        model = UNet3D(config)
        summary = model.get_model_summary()
        
        # Should use default values
        assert summary['input_channels'] == 1
        assert summary['output_channels'] == 3
        assert summary['depth'] == 4
        assert summary['initial_features'] == 32
        
        # Test forward pass with default configuration
        model.eval()  # Put in eval mode to avoid BatchNorm issues with small spatial dims
        input_tensor = torch.randn(1, 1, 16, 16, 16)  # Size that works with depth=4
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 3, 16, 16, 16), "Wrong output shape with minimal config"
    
    def test_odd_spatial_dimensions(self):
        """Test model with odd spatial dimensions."""
        config = {
            'model': {
                'input_channels': 1,
                'output_channels': 3,
                'depth': 2,  # Shallow to handle odd dimensions better
                'initial_features': 8,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.1,
                'weight_init': 'kaiming_normal'
            }
        }
        
        model = UNet3D(config)
        model.eval()
        
        # Test with odd dimensions
        input_tensor = torch.randn(1, 1, 9, 9, 9)  # Odd dimensions
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should have same spatial dimensions as input
        assert output.shape == (1, 3, 9, 9, 9), f"Wrong output shape: {output.shape}"
        assert torch.isfinite(output).all(), "Non-finite output with odd dimensions"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 