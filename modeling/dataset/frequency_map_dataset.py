"""
PyTorch Dataset class for Frequency Map Data (Gaussian Maps).

This module provides the FrequencyMapDataset class that loads frequency map data
from .npy files using memory mapping for efficient multiprocessing.
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
    
    This dataset loads frequency map data from .npy files using memory mapping.
    Memory mapping allows multiple worker processes to share the same physical
    memory pages, significantly reducing memory usage and improving performance
    with multiple DataLoader workers.
    
    Expected .npy structure:
    - Array shape: (N, 15, bins, bins)
    
    Args:
        npy_path: Path to the .npy file containing the data
        transform: Optional transform to apply to samples
        device: Device to place tensors on ('cpu', 'cuda', etc.)
    """
    
    def __init__(
        self, 
        npy_path: str, 
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the dataset with memory-mapped access.
        
        The data is loaded in read-only memory-mapped mode, which means:
        - Data is not loaded into memory until accessed
        - Multiple processes can share the same physical memory pages
        - Fast random access without full file decompression
        """
        self.npy_path = npy_path
        self.transform = transform
        self.device = device
        self.data_numpy = None  # Will be loaded lazily
        
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"{npy_path} not found.")
            
        # Load metadata only (shape) without keeping mmap open
        # This allows proper pickling for multiprocessing on Windows
        try:
            # Open mmap temporarily just to get shape, then close it
            temp_mmap = np.load(npy_path, mmap_mode='r')
            self.num_samples = temp_mmap.shape[0]
            self.shape = temp_mmap.shape[1:]  # (15, bins, bins)
            del temp_mmap  # Close the mmap immediately
            
            logger.info(f"Initialized FrequencyMapDataset: {self.num_samples} samples, shape {self.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset metadata from {npy_path}: {e}")
            raise e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def _ensure_mmap(self) -> None:
        """Open the memory-mapped array if not already open (per worker)."""
        if self.data_numpy is None:
            self.data_numpy = np.load(self.npy_path, mmap_mode='r')
    
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
        
        # Ensure memory-mapped array is open (lazy loading per worker)
        self._ensure_mmap()
            
        # Get data from memory-mapped array
        # This reads from disk/cache on demand
        sample = self.data_numpy[idx]
        
        # Convert to tensor
        # np.array() ensures we create a copy and detach from the mmap backing
        # This is necessary because torch.from_numpy() may not handle mmap arrays correctly
        # in all cases, and we want a proper float tensor copy
        tensor = torch.from_numpy(np.array(sample)).float()
        
        # Apply transforms if provided
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor
        
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a specific sample without loading the data fully.
        
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
            'path': self.npy_path
        }
        
        return info

    def get_data_statistics(self, max_samples: Optional[int] = 100) -> Dict[str, Any]:
        """
        Compute statistics for a subset of the dataset.
        
        For memory-mapped arrays, this will read the necessary data into memory
        for computation. If max_samples is specified, only a subset is loaded.
        
        Args:
            max_samples: Maximum number of samples to use for statistics (None for all)
            
        Returns:
            Dictionary with dataset statistics
        """
        n_samples = self.num_samples if max_samples is None else min(max_samples, self.num_samples)
        
        # Ensure memory-mapped array is open
        self._ensure_mmap()
        
        # For memory-mapped arrays, slicing will read the subset into memory
        if max_samples is not None and max_samples < self.num_samples:
            # Use evenly spaced subset
            indices = np.linspace(0, self.num_samples - 1, n_samples, dtype=int)
            data_subset = self.data_numpy[indices]
        else:
            data_subset = self.data_numpy
            
        # Compute statistics (this will trigger reads from mmap if needed)
        stats: Dict[str, Any] = {
            'mean': float(data_subset.mean()),
            'std': float(data_subset.std()),
            'min': float(data_subset.min()),
            'max': float(data_subset.max()),
            'samples_analyzed': n_samples,
            'total_samples': self.num_samples
        }
        
        return stats
    
    def __getstate__(self):
        """Ensure memory-mapped array is not pickled when using worker processes."""
        state = self.__dict__.copy()
        state['data_numpy'] = None  # Don't pickle the mmap
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling (in worker process)."""
        self.__dict__.update(state)
        # data_numpy will be lazily loaded when __getitem__ is called

