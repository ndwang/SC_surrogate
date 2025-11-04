"""
2D Variational Autoencoder (VAE) for multi-channel images.

This module implements a flexible 2D VAE architecture with an encoder-decoder
structure. The encoder downsamples using strided Conv2d blocks; the decoder
upsamples using Upsample + Conv2d blocks. Designed for inputs like 15-channel
64x64 images.
"""

from typing import Dict, Any, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .activations import get_activation

logger = logging.getLogger(__name__)


class EncoderBlock2D(nn.Module):
    """
    Encoder block for 2D inputs using Conv2d downsampling.

    Structure:
        [ReflectPad] -> Conv2d(stride=2 if downsample) -> [BN] -> Act -> [Dropout]
        -> [ReflectPad] -> Conv2d -> [BN] -> Act

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: ReflectPad2d padding size (0 disables pad wrapper)
        activation: Activation function name
        batch_norm: Whether to use BatchNorm2d
        dropout_rate: Dropout probability (0 disables)
        downsample: If True, first conv uses stride=2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        # Store activation as a module via shared utility
        self.activation = get_activation(activation)
        padding = kernel_size // 2

        # Convolve first (stride=1), then apply optional strided Conv2d downsampling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.downsample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
            if downsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.downsample_conv(x)
        return x


class DecoderBlock2D(nn.Module):
    """
    Decoder block for 2D inputs using Upsample + Conv2d.

    Structure:
        Upsample(x2) -> [ReflectPad] -> Conv2d -> [BN] -> Act -> [Dropout]
        -> [ReflectPad] -> Conv2d -> [BN] -> Act

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: ReflectPad2d padding size (0 disables pad wrapper)
        activation: Activation function name
        batch_norm: Whether to use BatchNorm2d
        dropout_rate: Dropout probability (0 disables)
        upsample_mode: 'bilinear' (default) or 'nearest'
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        upsample_mode: str = 'bilinear',
    ) -> None:
        super().__init__()

        # Store activation as a module via shared utility
        self.activation = get_activation(activation)
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False if upsample_mode == 'bilinear' else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolve first, then upsample
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x


class VAE2D(nn.Module):
    """
    2D Variational Autoencoder with Conv2d encoder and Upsample+Conv2d decoder.

    - Input: (batch, channels=15, H=64, W=64) by default
    - Encoder: series of EncoderBlock2D with strided Conv2d downsampling
    - Bottleneck: AdaptiveAvgPool2d -> Linear heads to mu and logvar
    - Decoder: Linear projection -> reshape -> series of DecoderBlock2D
    - Output: reconstructed input (same number of channels as input)

    Config fields (under config['model']):
        input_channels, hidden_channels, latent_dim, input_size, kernel_size,
        padding, activation, batch_norm, dropout_rate, weight_init,
        output_activation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get('model', {})

        self.input_channels = int(model_config.get('input_channels', 15))
        hidden_channels: List[int] = list(model_config.get('hidden_channels', [32, 64, 128]))
        self.latent_dim = int(model_config.get('latent_dim', 64))
        self.input_size = int(model_config.get('input_size', 64))  # assumes square input
        kernel_size = int(model_config.get('kernel_size', 3))
        activation = str(model_config.get('activation', 'relu'))
        batch_norm = bool(model_config.get('batch_norm', True))
        dropout_rate = float(model_config.get('dropout_rate', 0.0))
        self.weight_init = str(model_config.get('weight_init', 'kaiming_normal'))
        output_activation_name: Optional[str] = model_config.get('output_activation', None)

        if self.input_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError(
                f"input_size={self.input_size} is not divisible by 2**{len(hidden_channels)}; "
                f"downsampling would not land on integer spatial size."
            )

        # Encoder
        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = self.input_channels
        for out_ch in hidden_channels:
            block = EncoderBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                downsample=True,
            )
            self.encoder_blocks.append(block)
            in_ch = out_ch

        # Bottleneck heads (assume fixed input size; no global pooling)
        self.decoder_start_hw: int = self.input_size // (2 ** len(hidden_channels))
        bottleneck_features = hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw
        # Final projection to latent-dim space before parameter heads
        self.fc_bottleneck = nn.Linear(bottleneck_features, self.latent_dim)
        self.bottleneck_activation = get_activation(activation)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder projection
        self.fc_proj = nn.Linear(self.latent_dim, hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw)

        # Decoder blocks
        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        rev_channels = list(reversed(hidden_channels))
        for i in range(len(rev_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock2D(
                    in_channels=rev_channels[i],
                    out_channels=rev_channels[i + 1],
                    kernel_size=kernel_size,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    upsample_mode='bilinear',
                )
            )

        # Final conv to get back to input channels
        self.final_conv = nn.Conv2d(rev_channels[-1], self.input_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        if output_activation_name is None:
            self.output_activation = nn.Identity()
        elif output_activation_name == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation_name == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation_name}")

        # Initialize weights
        self._initialize_weights()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder path
        current = x
        for block in self.encoder_blocks:
            current = block(current)

        h = current.flatten(1)
        h = self.fc_bottleneck(h)
        h = self.bottleneck_activation(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_proj(z)
        h = h.view(z.size(0), -1, self.decoder_start_hw, self.decoder_start_hw)
        current = h
        for block in self.decoder_blocks:
            current = block(current)
        current = self.final_conv(current)
        return self.output_activation(current)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_model_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'VAE2D',
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'decoder_start_hw': self.decoder_start_hw,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }

    def _initialize_weights(self) -> None:
        """Initialize model weights according to selected method."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)


# Example usage and testing
if __name__ == "__main__":
    # Test model with sample config mirroring style in UNet3D
    test_config = {
        'model': {
            'input_channels': 15,
            'hidden_channels': [32, 64, 128],
            'latent_dim': 64,
            'input_size': 64,
            'kernel_size': 3,
            'padding': 1,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.0,
            'weight_init': 'kaiming_normal'
        }
    }

    model = VAE2D(test_config)
    print(f"Model summary: {model.get_model_summary()}")

    batch_size = 2
    x = torch.randn(batch_size, 15, 64, 64)
    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        recon, mu, logvar = model(x)
        print(f"Recon shape: {recon.shape}")
        print(f"Mu shape: {mu.shape}, LogVar shape: {logvar.shape}")

