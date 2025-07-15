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
from torch.utils.data import Dataset, DataLoader
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
        
        # Open HDF5 file and get dataset handles
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Get dataset handles (not data - for lazy loading)
        self.charge_density_ds = self.h5_file['charge_density']
        self.electric_field_ds = self.h5_file['electric_field']
        
        # Store dataset properties
        self.num_samples: int = int(self.charge_density_ds.shape[0])
        self.grid_shape = self.charge_density_ds.shape[1:]  # (Nx, Ny, Nz)
        
        # Validate dataset shapes
        expected_field_shape = (self.num_samples, 3, *self.grid_shape)
        if self.electric_field_ds.shape != expected_field_shape:
            raise ValueError(
                f"Electric field shape {self.electric_field_ds.shape} "
                f"doesn't match expected {expected_field_shape}"
            )
        
        logger.info(f"Loaded dataset: {self.num_samples} samples, grid shape {self.grid_shape}")
        logger.info(f"Charge density shape: {self.charge_density_ds.shape}")
        logger.info(f"Electric field shape: {self.electric_field_ds.shape}")
    
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
        
        # Lazy load data from HDF5
        charge_density = self.charge_density_ds[idx]  # shape: (Nx, Ny, Nz)
        electric_field = self.electric_field_ds[idx]  # shape: (3, Nx, Ny, Nz)
        
        # Convert to PyTorch tensors with correct dtype
        input_tensor = torch.tensor(charge_density, dtype=torch.float32)
        target_tensor = torch.tensor(electric_field, dtype=torch.float32)
        
        # Add channel dimension to input for CNN compatibility
        input_tensor = input_tensor.unsqueeze(0)  # shape: (1, Nx, Ny, Nz)
        
        # Move to specified device
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        # Apply transforms if provided
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return input_tensor, target_tensor
    
    def __del__(self) -> None:
        """Clean up HDF5 file handle."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
    
    def close(self) -> None:
        """Explicitly close the HDF5 file."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
            self.h5_file = None
    
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
        
        # Get basic shape information
        info: Dict[str, Any] = {
            'index': idx,
            'charge_density_shape': self.charge_density_ds[idx:idx+1].shape[1:],
            'electric_field_shape': self.electric_field_ds[idx:idx+1].shape[1:],
            'grid_shape': self.grid_shape
        }
        
        # Add metadata from HDF5 attributes if available
        if hasattr(self.h5_file, 'attrs'):
            for key, value in self.h5_file.attrs.items():
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
        
        for idx in indices:
            charge_density = self.charge_density_ds[idx]
            electric_field = self.electric_field_ds[idx]
            
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


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = 'cpu'
) -> Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], DataLoader[Tuple[torch.Tensor, torch.Tensor]], DataLoader[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Create DataLoaders for train, validation, and test datasets.
    
    Args:
        train_path: Path to train.h5
        val_path: Path to val.h5
        test_path: Path to test.h5
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        device: Device to place tensors on
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SpaceChargeDataset(train_path, device=device)
    val_dataset = SpaceChargeDataset(val_path, device=device)
    test_dataset = SpaceChargeDataset(test_path, device=device)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Basic functionality test
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = "data/processed/train.h5"
    
    print(f"Testing SpaceChargeDataset with {h5_path}")
    
    try:
        # Create dataset
        dataset = SpaceChargeDataset(h5_path)
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Grid shape: {dataset.grid_shape}")
        
        # Test sample loading
        input_tensor, target_tensor = dataset[0]
        print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Target shape: {target_tensor.shape}, dtype: {target_tensor.dtype}")
        
        # Test with DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch_input, batch_target = next(iter(loader))
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        
        # Get statistics
        stats = dataset.get_data_statistics(max_samples=10)
        print(f"Dataset statistics: {stats}")
        
        dataset.close()
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        raise 