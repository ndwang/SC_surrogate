"""
Loss functions for Space Charge Surrogate Model.

This module defines standard and custom loss functions, and provides a registry
and factory for extensible loss selection and composition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Any
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