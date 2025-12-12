"""
PyTorch Dataset class for Frequency Map Data (Gaussian Maps).

This module provides the FrequencyMapDataset class that loads frequency map data
from NPZ files for VAE training.
"""

import os
import torch
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, Union
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class FrequencyMapDataset(Dataset[torch.Tensor]):
    """
    PyTorch Dataset for loading frequency map data (Gaussian maps).
    
    This dataset loads frequency map data from NPZ files.
    
    Expected NPZ structure:
    - 'data': shape (N, 15, bins, bins)
    
    Args:
        npz_path: Path to the .npz file containing the data
        transform: Optional transform to apply to samples
        device: Device to place tensors on ('cpu', 'cuda', etc.)
    """
    
    def __init__(
        self, 
        npz_path: str, 
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: str = 'cpu'
    ) -> None:
        """Initialize the dataset."""
        self.npz_path = npz_path
        self.transform = transform
        self.device = device
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"{npz_path} not found.")
            
        # Load data into memory (assuming it fits, as in original implementation)
        # For very large datasets, we might want to switch to lazy loading or memmap
        try:
            npzfile = np.load(npz_path)
            if 'data' not in npzfile:
                 raise KeyError(f"Key 'data' not found in {npz_path}. Available keys: {list(npzfile.keys())}")
            
            self.data_numpy = npzfile["data"]  # shape (N, 15, bins, bins)
            self.num_samples = self.data_numpy.shape[0]
            self.shape = self.data_numpy.shape[1:] # (15, bins, bins)
            
            logger.info(f"Loaded FrequencyMapDataset: {self.num_samples} samples, shape {self.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset from {npz_path}: {e}")
            raise e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            tensor: frequency maps with shape (15, bins, bins)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
            
        # Get data
        sample = self.data_numpy[idx]
        
        # Convert to tensor
        tensor = torch.from_numpy(sample).float()
        
        # Apply transforms if provided
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor
        
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a specific sample without loading the data fully (if lazy).
        Since we load all data in init, this is just metadata access.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample metadata
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
            
        info: Dict[str, Any] = {
            'index': idx,
            'shape': self.shape,
            'path': self.npz_path
        }
        
        return info

    def get_data_statistics(self, max_samples: Optional[int] = 100) -> Dict[str, Any]:
        """
        Compute statistics for a subset of the dataset.
        
        Args:
            max_samples: Maximum number of samples to use for statistics (None for all)
            
        Returns:
            Dictionary with dataset statistics
        """
        n_samples = self.num_samples if max_samples is None else min(max_samples, self.num_samples)
        
        # If we have all data in memory, we can compute stats easily
        # If max_samples is small, we slice
        if max_samples is not None and max_samples < self.num_samples:
            # Use random subset or first n samples
            indices = np.linspace(0, self.num_samples - 1, n_samples, dtype=int)
            data_subset = self.data_numpy[indices]
        else:
            data_subset = self.data_numpy
            
        stats: Dict[str, Any] = {
            'mean': float(data_subset.mean()),
            'std': float(data_subset.std()),
            'min': float(data_subset.min()),
            'max': float(data_subset.max()),
            'samples_analyzed': n_samples,
            'total_samples': self.num_samples
        }
        
        return stats

