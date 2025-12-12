"""
PyTorch Dataset class for Space Charge Surrogate Model.

This module provides the SpaceChargeDataset class that efficiently loads
processed HDF5 data for PyTorch model training with proper lazy loading
and tensor shape handling for CNN compatibility.

Key features:
- Lazy loading from HDF5 files for memory efficiency
- Automatic conversion to PyTorch tensors with correct dtypes
- Proper tensor shapes for CNN: input (1, Nx, Ny, Nz), target (3, Nx, Ny, Nz)
- Support for DataLoader integration
"""

import h5py
import torch
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class SpaceChargeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for loading preprocessed space charge simulation data.
    
    This dataset loads charge density (input) and electric field (target) data
    from processed HDF5 files. The data is lazy-loaded for memory efficiency
    and automatically converted to appropriate PyTorch tensor formats.
    
    Expected HDF5 structure:
    - 'charge_density': shape (N, Nx, Ny, Nz), normalized charge density data
    - 'electric_field': shape (N, 3, Nx, Ny, Nz), normalized electric field data
    
    Args:
        h5_path: Path to processed HDF5 file (train.h5, val.h5, or test.h5)
        transform: Optional transform to apply to samples
        device: Device to place tensors on ('cpu', 'cuda', etc.)
    """
    
    def __init__(
        self, 
        h5_path: str, 
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        device: str = 'cpu'
    ) -> None:
        """Initialize the dataset."""
        self.h5_path = h5_path
        self.transform = transform
        self.device = device
        self._h5 = None  # Will be opened lazily per worker
        
        # Open file once to get shape info, then close
        with h5py.File(h5_path, 'r') as h5_file:
            self.num_samples: int = int(h5_file['charge_density'].shape[0])
            self.grid_shape = h5_file['charge_density'].shape[1:]  # (Nx, Ny, Nz)
            expected_field_shape = (self.num_samples, 3, *self.grid_shape)
            if h5_file['electric_field'].shape != expected_field_shape:
                raise ValueError(
                    f"Electric field shape {h5_file['electric_field'].shape} "
                    f"doesn't match expected {expected_field_shape}"
                )
        
        logger.info(f"Loaded dataset: {self.num_samples} samples, grid shape {self.grid_shape}")
        logger.info("Charge density shape: (N, Nx, Ny, Nz)")
        logger.info("Electric field shape: (N, 3, Nx, Ny, Nz)")

    def _ensure_open(self) -> None:
        """Open the HDF5 file handle if not already open (per worker)."""
        if self._h5 is None:
            # libver='latest' can improve I/O performance; SWMR not needed for read-only single-writer
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest')
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_tensor, target_tensor) where:
            - input_tensor: charge density with shape (1, Nx, Ny, Nz)
            - target_tensor: electric field with shape (3, Nx, Ny, Nz)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        self._ensure_open()
        # Lazy load data from HDF5 (numpy arrays)
        charge_density = self._h5['charge_density'][idx]  # shape: (Nx, Ny, Nz)
        electric_field = self._h5['electric_field'][idx]  # shape: (3, Nx, Ny, Nz)
        
        # Convert to PyTorch tensors efficiently on CPU
        input_tensor = torch.from_numpy(charge_density).float()
        target_tensor = torch.from_numpy(electric_field).float()
        
        # Add channel dimension to input for CNN compatibility
        input_tensor = input_tensor.unsqueeze(0)  # shape: (1, Nx, Ny, Nz)
        
        # Apply transforms if provided
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return input_tensor, target_tensor
    
    def __getstate__(self):
        """Ensure file handle is not pickled when using worker processes."""
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __del__(self) -> None:
        """Clean up HDF5 file handle."""
        try:
            if getattr(self, '_h5', None) is not None:
                self._h5.close()
                self._h5 = None
        except Exception:
            pass
    
    def close(self) -> None:
        """Explicitly close the HDF5 file."""
        pass # No persistent file handle to close
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a specific sample without loading the data.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample metadata
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        with h5py.File(self.h5_path, 'r') as h5_file:
            # Get basic shape information
            info: Dict[str, Any] = {
                'index': idx,
                'charge_density_shape': h5_file['charge_density'][idx:idx+1].shape[1:],
                'electric_field_shape': h5_file['electric_field'][idx:idx+1].shape[1:],
                'grid_shape': self.grid_shape
            }
            
            # Add metadata from HDF5 attributes if available
            if hasattr(h5_file, 'attrs'):
                for key, value in h5_file.attrs.items():
                    info[f'dataset_{key}'] = value
        
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
        indices = np.linspace(0, self.num_samples - 1, n_samples, dtype=int)
        
        charge_values = []
        field_values = []
        
        with h5py.File(self.h5_path, 'r') as h5_file:
            for idx in indices:
                charge_density = h5_file['charge_density'][idx]
                electric_field = h5_file['electric_field'][idx]
                
                charge_values.append(charge_density.flatten())
                field_values.append(electric_field.flatten())
        
        charge_all = np.concatenate(charge_values)
        field_all = np.concatenate(field_values)
        
        stats: Dict[str, Any] = {
            'charge_density': {
                'mean': float(charge_all.mean()),
                'std': float(charge_all.std()),
                'min': float(charge_all.min()),
                'max': float(charge_all.max())
            },
            'electric_field': {
                'mean': float(field_all.mean()),
                'std': float(field_all.std()),
                'min': float(field_all.min()),
                'max': float(field_all.max())
            },
            'samples_analyzed': n_samples,
            'total_samples': self.num_samples
        }
        
        return stats

