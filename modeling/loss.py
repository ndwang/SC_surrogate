"""
Loss functions for Space Charge Surrogate Model.

This module defines standard and custom loss functions, and provides a registry
and factory for extensible loss selection and composition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Any, Tuple
import numpy as np
from preprocessing.scalers import SymlogScaler

# Registry for loss functions
LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_loss(name: str):
    """Decorator to register a loss function by name."""
    def decorator(cls_or_fn):
        LOSS_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn
    return decorator

@register_loss('mse')
def mse_loss(**kwargs) -> nn.Module:
    return nn.MSELoss(**kwargs)

@register_loss('l1')
@register_loss('mae')
def l1_loss(**kwargs) -> nn.Module:
    return nn.L1Loss(**kwargs)

@register_loss('huber')
def huber_loss(**kwargs) -> nn.Module:
    return nn.HuberLoss(**kwargs)

@register_loss('kl_regularization')
class KLRegularizationLoss(nn.Module):
    """
    KL Regularization loss for Variational Autoencoders.
    
    Computes KL(q(z|x) || p(z)) where:
    - q(z|x) ~ N(mu, exp(logvar)) is the learned posterior
    - p(z) ~ N(0, I) is the standard normal prior
    
    The loss is computed as:
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    Args:
        reduction: 'mean' (default) or 'sum' - how to aggregate the loss across batch
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL Regularization loss.
        
        Args:
            mu: Mean of the posterior distribution, shape (batch, latent_dim)
            logvar: Log variance of the posterior distribution, shape (batch, latent_dim)
            
        Returns:
            Scalar loss value
        """
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)
        
        return kl

@register_loss('kl_divergence')
class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for comparing two normalized density distributions.
    
    This is different from the KL regularization term for VAEs. This loss compares
    two probability distributions (normalized densities) P and Q.
    
    Computes KL(P || Q) = sum(P * log(P / Q)) where:
    - P is the target (ground truth) normalized density
    - Q is the predicted normalized density
    
    The inputs should be normalized probability distributions (non-negative and sum to 1).
    
    Args:
        eps: Small epsilon value to avoid numerical issues with log(0). Default: 1e-8
        reduction: 'mean' (default) or 'sum' or 'none' - how to aggregate across batch
        normalize: If True, normalize inputs to ensure they sum to 1. Default: True
        per_channel: If True, compute KL divergence separately for each channel and aggregate.
                     If False, treat all dimensions (including channels) as one distribution.
                     Default: False
        channel_reduction: 'mean' (default) or 'sum' - how to aggregate across channels
                           when per_channel=True. Ignored if per_channel=False.
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean', normalize: bool = True,
                 per_channel: bool = False, channel_reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        if channel_reduction not in ['mean', 'sum']:
            raise ValueError(f"channel_reduction must be 'mean' or 'sum', got {channel_reduction}")
        self.eps = eps
        self.reduction = reduction
        self.normalize = normalize
        self.per_channel = per_channel
        self.channel_reduction = channel_reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between predicted and target normalized densities.
        
        Args:
            pred: Predicted normalized density, shape (batch, ...) or (batch, channels, ...)
            target: Target (ground truth) normalized density, shape (batch, ...) or (batch, channels, ...)
            
        Returns:
            Scalar loss value (or tensor if reduction='none')
        """
        # Ensure non-negative
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        
        if self.per_channel and pred.dim() >= 3:
            # Per-channel mode: normalize and compute KL separately for each channel
            # Expected shape: (batch, channels, ...)
            batch_size = pred.size(0)
            num_channels = pred.size(1)
            
            # Flatten spatial dimensions: (batch, channels, spatial_dims...)
            # -> (batch, channels, spatial_size)
            spatial_dims = pred.shape[2:]
            spatial_size = int(np.prod(spatial_dims))
            pred_flat = pred.view(batch_size, num_channels, spatial_size)
            target_flat = target.view(batch_size, num_channels, spatial_size)
            
            # Normalize each channel separately if requested
            if self.normalize:
                # Normalize over spatial dimensions for each channel
                pred_norm = pred_flat / (pred_flat.sum(dim=2, keepdim=True) + self.eps)
                target_norm = target_flat / (target_flat.sum(dim=2, keepdim=True) + self.eps)
            else:
                pred_norm = pred_flat
                target_norm = target_flat
            
            # Compute KL divergence per channel: (batch, channels)
            kl_per_channel = target_norm * (
                torch.log(target_norm + self.eps) - torch.log(pred_norm + self.eps)
            )
            kl_per_channel = kl_per_channel.sum(dim=2)  # Sum over spatial dimensions
            
            # Aggregate across channels
            if self.channel_reduction == 'mean':
                kl = torch.mean(kl_per_channel, dim=1)  # (batch,)
            else:  # 'sum'
                kl = torch.sum(kl_per_channel, dim=1)  # (batch,)
        else:
            # Global mode: treat all dimensions (including channels) as one distribution
            # Normalize if requested
            if self.normalize:
                # Flatten all dimensions except batch for normalization
                pred_flat = pred.view(pred.size(0), -1)
                target_flat = target.view(target.size(0), -1)
                
                # Normalize to sum to 1
                pred_norm = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + self.eps)
                target_norm = target_flat / (target_flat.sum(dim=1, keepdim=True) + self.eps)
                
                # Reshape back to original shape
                pred = pred_norm.view_as(pred)
                target = target_norm.view_as(target)
            
            # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
            kl = target * (torch.log(target + self.eps) - torch.log(pred + self.eps))
            
            # Sum over all dimensions except batch dimension
            kl = kl.view(kl.size(0), -1).sum(dim=1)  # (batch,)
        
        # Apply batch reduction
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)
        # If reduction == 'none', return per-sample KL divergences (batch,)
        
        return kl

class BetaVAELoss(nn.Module):
    """
    Beta-VAE loss function combining reconstruction loss and KL regularization.
    
    The beta-VAE loss is defined as:
    Loss = Reconstruction_Loss + β * KL_Regularization
    
    Where:
    - Reconstruction_Loss: Loss between reconstructed output and target (e.g., MSE, KL divergence)
    - KL_Regularization: KL divergence between learned posterior q(z|x) and prior p(z) = N(0, I)
    - β: Weight parameter controlling the trade-off between reconstruction quality and 
         latent space regularization (higher β encourages more disentangled representations)
    
    Args:
        reconstruction_loss_fn: Initialized reconstruction loss module
        kl_loss_fn: Initialized KL regularization loss module
        beta: Weight for KL regularization term. Default: 1.0 (standard VAE)
    """
    
    def __init__(
        self,
        reconstruction_loss_fn: nn.Module,
        kl_loss_fn: nn.Module,
        beta: float = 1.0
    ):
        super().__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.beta = float(beta)
    
    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute beta-VAE loss.
        
        Args:
            outputs: Tuple of (reconstructed, mu, logvar) from VAE model
                     - reconstructed: Reconstructed output, shape (batch, channels, H, W)
                     - mu: Mean of posterior distribution, shape (batch, latent_dim)
                     - logvar: Log variance of posterior distribution, shape (batch, latent_dim)
            targets: Ground truth target data, shape (batch, channels, H, W)
            
        Returns:
            Tuple containing:
            - total_loss: Scalar loss value (reconstruction_loss + beta * kl_loss)
            - metrics: Dictionary containing 'recon_loss' and 'kl_loss' tensors
        """
        recon, mu, logvar = outputs
        
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss_fn(recon, targets)
        
        # Compute KL regularization loss
        kl_loss = self.kl_loss_fn(mu, logvar)
        
        # Combine losses
        total_loss = recon_loss + self.beta * kl_loss
        
        metrics = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }
        
        return total_loss, metrics

@register_loss('beta_vae')
def beta_vae_loss(**kwargs) -> nn.Module:
    """
    Factory function to create a BetaVAELoss from configs.
    
    When called from get_loss_from_config with a dict config, extracts:
    - reconstruction_loss: Config for reconstruction loss (string or dict with 'type' and params).
                          Default: 'mse'
    - kl_regularization: Config for KL regularization loss (string or dict with 'type' and params).
                        If None, uses default KLRegularizationLoss with reduction='mean'.
                        Default: None
    - beta: Weight for KL regularization term. Default: 1.0 (standard VAE)
    
    Returns:
        BetaVAELoss instance
    """
    # Extract beta_vae-specific arguments from kwargs
    reconstruction_loss_config = kwargs.pop('reconstruction_loss', 'mse')
    kl_regularization_config = kwargs.pop('kl_regularization', None)
    beta = kwargs.pop('beta', 1.0)

    # Initialize reconstruction loss from config
    recon_loss_fn = get_loss_from_config(reconstruction_loss_config)
    
    # Initialize KL regularization loss from config
    if kl_regularization_config is None:
        kl_loss_fn = KLRegularizationLoss(reduction='mean')
    else:
        kl_loss_fn = get_loss_from_config(kl_regularization_config)
    
    return BetaVAELoss(
        reconstruction_loss_fn=recon_loss_fn,
        kl_loss_fn=kl_loss_fn,
        beta=beta
    )

# Example: Combined loss (weighted sum)
class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = losses
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            self.weights = weights
    def forward(self, input, target):
        return sum(w * loss(input, target) for w, loss in zip(self.weights, self.losses))

@register_loss('combined')
def combined_loss(loss_configs, weights=None, **kwargs) -> nn.Module:
    # loss_configs: list of dicts, each with 'type' and optional params
    losses = [get_loss_from_config(cfg) for cfg in loss_configs]
    return CombinedLoss(losses, weights)

def get_loss_from_config(config: Any) -> nn.Module:
    """
    Create a loss function from config.
    config can be a string (loss name) or dict with 'type' and params.
    """
    if isinstance(config, str):
        loss_type = config.lower()
        if loss_type not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss function: {loss_type}")
        return LOSS_REGISTRY[loss_type]()
    elif isinstance(config, dict):
        loss_type = config.get('type', 'mse').lower()
        params = {k: v for k, v in config.items() if k != 'type'}
        if loss_type == 'combined':
            # Special handling for combined loss
            loss_configs = config.get('losses', [])
            weights = config.get('weights', None)
            return LOSS_REGISTRY['combined'](loss_configs, weights, **params)
        if loss_type not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss function: {loss_type}")
        return LOSS_REGISTRY[loss_type](**params)
    else:
        raise TypeError(f"Invalid loss config: {config}") 