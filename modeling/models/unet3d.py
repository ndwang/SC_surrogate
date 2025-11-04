"""
3D U-Net model for Space Charge Surrogate modeling.

This module implements a flexible 3D U-Net architecture with encoder-decoder
structure and skip connections. The model is well-suited for volumetric data
tasks such as regression on 3D grids.

Based on the original U-Net paper: https://arxiv.org/abs/1606.06650
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import logging
from .activations import get_activation

logger = logging.getLogger(__name__)


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net: Conv3d -> BatchNorm -> Activation -> Dropout -> MaxPool.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Convolution kernel size
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout probability
        activation: Activation function name
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        super(EncoderBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            bias=not use_batch_norm
        )
        
        self.batch_norm = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.pool = nn.MaxPool3d(2, stride=2)
        self.activation = get_activation(activation)
    
    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (pooled_output, skip_connection): Tuple of pooled output and features for skip connection
        """
        # Convolution, normalization, activation, dropout
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        skip = x.clone()  # Save for skip connection before dropout and pooling
        x = self.dropout(x)
        
        # Pooling for next level
        pooled = self.pool(x)
        
        return pooled, skip


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net: ConvTranspose -> Concat -> Conv -> BatchNorm -> Activation -> Dropout.
    
    Args:
        in_channels: Number of input channels (from previous decoder level)
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout probability
        activation: Activation function name
    """
    
    def __init__(
        self, 
        in_channels: int,
        skip_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        super(DecoderBlock, self).__init__()
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, 
            kernel_size=2, 
            stride=2
        )
        
        # Convolution after concatenation
        concat_channels = in_channels + skip_channels
        self.conv = nn.Conv3d(
            concat_channels, out_channels,
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            bias=not use_batch_norm
        )
        
        self.batch_norm = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.activation = get_activation(activation)
    
    
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor from previous decoder level
            skip: Skip connection tensor from encoder
            
        Returns:
            Output tensor after upsampling, concatenation, and convolution
        """
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch between upsampled and skip features
        # This can happen due to odd spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            # Interpolate x to match skip dimensions
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        
        # Convolution, normalization, activation, dropout
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric data processing.
    
    This model implements a U-Net architecture with configurable depth and features.
    It consists of an encoder (downsampling) path, bottleneck, and decoder (upsampling)
    path with skip connections between corresponding encoder and decoder levels.
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the UNet3D model."""
        super(UNet3D, self).__init__()
        
        # Extract model configuration
        model_config = config.get('model', {})
        
        self.input_channels = model_config.get('input_channels', 1)
        self.output_channels = model_config.get('output_channels', 3)
        self.depth = model_config.get('depth', 4)  # Number of encoder/decoder levels
        self.initial_features = model_config.get('initial_features', 32)
        self.kernel_size = model_config.get('kernel_size', 3)
        self.activation = model_config.get('activation', 'relu')
        self.use_batch_norm = model_config.get('batch_norm', True)
        self.dropout_rate = model_config.get('dropout_rate', 0.1)
        self.weight_init = model_config.get('weight_init', 'kaiming_normal')
        
        # Build encoder path
        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = self.input_channels
        
        for i in range(self.depth):
            out_ch = self.initial_features * (2 ** i)  # Double features at each level
            block = EncoderBlock(
                in_ch, out_ch,
                kernel_size=self.kernel_size,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            )
            self.encoder_blocks.append(block)
            in_ch = out_ch
        
        # Bottleneck (no pooling)
        bottleneck_channels = self.initial_features * (2 ** self.depth)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_ch, bottleneck_channels, 
                     kernel_size=self.kernel_size, 
                     padding=self.kernel_size//2,
                     bias=not self.use_batch_norm),
            nn.BatchNorm3d(bottleneck_channels) if self.use_batch_norm else nn.Identity(),
            nn.ReLU() if self.activation.lower() == 'relu' else nn.Identity(),  # type: ignore
            nn.Dropout3d(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
        
        # Build decoder path
        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = bottleneck_channels
        
        for i in range(self.depth):
            # Skip connection comes from encoder level (depth - 1 - i)
            skip_ch = self.initial_features * (2 ** (self.depth - 1 - i))
            out_ch = skip_ch  # Output same channels as skip connection
            
            block = DecoderBlock(
                in_ch, skip_ch, out_ch,
                kernel_size=self.kernel_size,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            )
            self.decoder_blocks.append(block)  # type: ignore
            in_ch = out_ch
        
        # Final output layer
        self.final_conv = nn.Conv3d(
            in_ch, self.output_channels,
            kernel_size=1,  # 1x1x1 conv for final output
            bias=True
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"UNet3D initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Architecture: {self.input_channels} -> depth={self.depth}, features={self.initial_features} -> {self.output_channels}")
    
    def _initialize_weights(self) -> None:
        """Initialize model weights according to the specified method."""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, D, H, W)
            
        Returns:
            Output tensor of shape (batch_size, output_channels, D, H, W)
        """
        # Store skip connections
        skip_connections: List[torch.Tensor] = []
        
        # Encoder path
        current = x
        for encoder_block in self.encoder_blocks:
            current, skip = encoder_block(current)
            skip_connections.append(skip)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path (reverse order of skip connections)
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]  # Get skip connection in reverse order
            current = decoder_block(current, skip)
        
        # Final output
        output = self.final_conv(current)
        
        return output
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNet3D',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'depth': self.depth,
            'initial_features': self.initial_features,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
            'weight_init': self.weight_init,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_blocks': len(self.encoder_blocks),
            'decoder_blocks': len(self.decoder_blocks)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test model with sample config
    test_config = {
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
    
    # Create model
    model = UNet3D(test_config)
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
    
    print("UNet3D test completed successfully!") 