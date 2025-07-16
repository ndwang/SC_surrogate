"""
3D CNN model for Space Charge Surrogate modeling.

This module implements a flexible 3D CNN architecture that takes charge density
as input and predicts electric field components. The model architecture is
configurable through the training config file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CNN3D(nn.Module):
    """
    3D Convolutional Neural Network for space charge surrogate modeling.
    
    This model takes 3D charge density data as input and predicts 3D electric
    field components. The architecture consists of encoder-decoder style
    convolutional layers with residual connections.
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the CNN3D model."""
        super(CNN3D, self).__init__()
        
        # Extract model configuration
        model_config = config.get('model', {})
        
        self.input_channels = model_config.get('input_channels', 1)
        self.output_channels = model_config.get('output_channels', 3)
        self.hidden_channels = model_config.get('hidden_channels', [32, 64, 128, 64, 32])
        self.kernel_size = model_config.get('kernel_size', 3)
        self.padding = model_config.get('padding', 1)
        self.activation = model_config.get('activation', 'relu')
        self.use_batch_norm = model_config.get('batch_norm', True)
        self.dropout_rate = model_config.get('dropout_rate', 0.1)
        self.weight_init = model_config.get('weight_init', 'kaiming_normal')
        
        # Build the network layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Create encoder path (increasing channels)
        all_channels = [self.input_channels] + self.hidden_channels
        encoder_channels = all_channels[:len(all_channels)//2 + 1]  # First half + middle
        
        for i in range(len(encoder_channels) - 1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            
            # Convolutional layer
            conv_layer = nn.Conv3d(
                in_ch, out_ch, 
                kernel_size=self.kernel_size,
                padding=0,  # Use manual reflection padding
                bias=not self.use_batch_norm
            )
            self.encoder_layers.append(conv_layer)
            
            # Batch normalization
            if self.use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm3d(out_ch))
            else:
                self.batch_norm_layers.append(nn.Identity())
            
            # Dropout
            if self.dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout3d(self.dropout_rate))
            else:
                self.dropout_layers.append(nn.Identity())
        
        # Create decoder path (decreasing channels to output)
        decoder_channels = encoder_channels[-1:] + all_channels[len(all_channels)//2 + 1:] + [self.output_channels]
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Convolutional layer
            conv_layer = nn.Conv3d(
                in_ch, out_ch,
                kernel_size=self.kernel_size,
                padding=0,  # Use manual reflection padding
                bias=not self.use_batch_norm or i == len(decoder_channels) - 2  # No bias for output layer
            )
            self.decoder_layers.append(conv_layer)
            
            # No batch norm or dropout for the final output layer
            if i < len(decoder_channels) - 2:
                if self.use_batch_norm:
                    self.batch_norm_layers.append(nn.BatchNorm3d(out_ch))
                else:
                    self.batch_norm_layers.append(nn.Identity())
                
                if self.dropout_rate > 0:
                    self.dropout_layers.append(nn.Dropout3d(self.dropout_rate))
                else:
                    self.dropout_layers.append(nn.Identity())
        
        # Initialize weights
        self._initialize_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CNN3D initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Architecture: {self.input_channels} -> {' -> '.join(map(str, self.hidden_channels))} -> {self.output_channels}")
    
    def _initialize_weights(self) -> None:
        """Initialize model weights according to the specified method."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                if self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # Default PyTorch initialization
                    pass
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _get_activation(self) -> nn.Module:
        """Get the activation function based on config."""
        if self.activation.lower() == 'relu':
            return F.relu
        elif self.activation.lower() == 'leaky_relu':
            return F.leaky_relu
        elif self.activation.lower() == 'elu':
            return F.elu
        elif self.activation.lower() == 'gelu':
            return F.gelu
        else:
            logger.warning(f"Unknown activation {self.activation}, using ReLU")
            return F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, Nx, Ny, Nz)
            
        Returns:
            Output tensor of shape (batch_size, 3, Nx, Ny, Nz)
        """
        activation_fn = self._get_activation()
        
        # Store intermediate features for potential residual connections
        encoder_features = []
        
        # Encoder path
        current = x
        bn_idx = 0
        dropout_idx = 0
        pad = self.padding
        pad_tuple = (pad, pad, pad, pad, pad, pad)  # (D, D, H, H, W, W)
        for conv_layer in self.encoder_layers:
            current = F.pad(current, pad_tuple, mode='reflect')
            current = conv_layer(current)
            current = self.batch_norm_layers[bn_idx](current)
            current = activation_fn(current)
            current = self.dropout_layers[dropout_idx](current)
            
            encoder_features.append(current)
            bn_idx += 1
            dropout_idx += 1
        
        # Decoder path
        for i, conv_layer in enumerate(self.decoder_layers):
            current = F.pad(current, pad_tuple, mode='reflect')
            current = conv_layer(current)
            
            # Apply batch norm and activation for all layers except the last
            if i < len(self.decoder_layers) - 1:
                current = self.batch_norm_layers[bn_idx](current)
                current = activation_fn(current)
                current = self.dropout_layers[dropout_idx](current)
                bn_idx += 1
                dropout_idx += 1
        
        return current
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CNN3D',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': self.activation,
            'batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
            'weight_init': self.weight_init,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_layers': len(self.encoder_layers),
            'decoder_layers': len(self.decoder_layers)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test model with sample config
    test_config = {
        'model': {
            'input_channels': 1,
            'output_channels': 3,
            'hidden_channels': [32, 64, 128, 64, 32],
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
    print(f"Model summary: {model.get_model_summary()}")
    
    # Test forward pass
    batch_size = 2
    grid_size = (16, 16, 16)
    
    # Create dummy input
    x = torch.randn(batch_size, 1, *grid_size)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
    
    print("CNN3D test completed successfully!") 