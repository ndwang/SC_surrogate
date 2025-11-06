"""
Common activation utilities for model modules.
"""

from typing import Optional
import torch.nn as nn

def get_activation(name: Optional[str] = None) -> nn.Module:
    """Return an nn.Module activation by name.

    Supported activations:
        - 'relu' (default if name is empty/whitespace)
        - 'leaky_relu'
        - 'elu'
        - 'gelu'
        - 'sigmoid'
        - 'tanh'
        - None (returns Identity)

    Args:
        name: Activation name. If None, returns Identity. If empty/whitespace, defaults to 'relu'.

    Returns:
        nn.Module: The activation module
    """
    if name is None:
        return nn.Identity()
    
    n = name.strip().lower() if name else ''
    if not n:
        return nn.ReLU()
    
    if n == 'relu':
        return nn.ReLU()
    if n == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)
    if n == 'elu':
        return nn.ELU()
    if n == 'gelu':
        return nn.GELU()
    if n == 'sigmoid':
        return nn.Sigmoid()
    if n == 'tanh':
        return nn.Tanh()
    
    raise ValueError(f"Unsupported activation: {name}")

