"""
Common activation utilities for model modules.
"""

import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    """Return an nn.Module activation by name.

    Supported: 'relu', 'leaky_relu', 'elu', 'gelu'. Defaults to ReLU.
    """
    n = (name or 'relu').lower().strip()
    if n == 'relu':
        return nn.ReLU()
    if n == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)
    if n == 'elu':
        return nn.ELU()
    if n == 'gelu':
        return nn.GELU()
    return nn.ReLU()

