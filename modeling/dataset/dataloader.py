"""
Data loader creation utilities and dataset registry.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any, Union, Type
import logging

from .space_charge_dataset import SpaceChargeDataset
from .frequency_map_dataset import FrequencyMapDataset

logger = logging.getLogger(__name__)

# Registry for dataset types
DATASET_REGISTRY = {
    "space_charge": SpaceChargeDataset,
    "frequency_map": FrequencyMapDataset,
}

def register_dataset_type(type_name: str, dataset_class: Type):
    """
    Register a new dataset type.
    
    Args:
        type_name: String identifier for the dataset type
        dataset_class: Class definition for the dataset
    """
    DATASET_REGISTRY[type_name] = dataset_class
    logger.info(f"Registered dataset type: {type_name}")

def create_data_loaders(
    data_path: str,
    dataset_type: str = "space_charge",
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cpu',
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for a dataset.
    
    Supports different dataset types via the dataset_type parameter.
    For datasets that require train/val/test splits (like space_charge),
    call this function three times with different paths.
    
    Args:
        data_path: Path to the data file
        dataset_type: Type of dataset to create ("space_charge" or "frequency_map")
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        device: Device to place tensors on
        shuffle: Whether to shuffle the data (default: True for training)
        drop_last: Whether to drop the last incomplete batch (default: True)
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        DataLoader instance
        
    Example:
        # For space_charge with train/val/test splits:
        train_loader = create_data_loaders("train.h5", shuffle=True)
        val_loader = create_data_loaders("val.h5", shuffle=False)
        test_loader = create_data_loaders("test.h5", shuffle=False)
        
        # For frequency_map (single file):
        loader = create_data_loaders("frequency_maps.npy", dataset_type="frequency_map")
    """
    
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}")
        
    DatasetClass = DATASET_REGISTRY[dataset_type]
    
    # Create dataset
    dataset = DatasetClass(data_path, device=device, **kwargs)
    
    # Performance-friendly flags
    persistent = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None
    
    # Use fewer workers for validation/test (when shuffle=False)
    # This is a heuristic - can be overridden by caller if needed
    effective_num_workers = num_workers
    if not shuffle and num_workers > 0:
        effective_num_workers = max(1, num_workers // 2)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )
    
    logger.info(f"Created DataLoader for {dataset_type} with batch_size={batch_size}, shuffle={shuffle}")
    logger.info(f"Dataset: {len(dataset)} samples, {len(loader)} batches")
    
    return loader

