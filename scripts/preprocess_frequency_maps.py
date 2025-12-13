import os
import numpy as np
import yaml
import logging
import argparse
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_frequency_maps(config_path: str):
    """
    Split raw frequency maps into train/val/test sets.
    """
    config = load_config(config_path)
    
    # Paths
    raw_path = config['paths']['raw_data_path']
    processed_dir = config['paths']['processed_dir']
    
    # Preprocessing config
    split_ratios = config['preprocessing']['split_ratios']
    seed = config['preprocessing']['split_seed']
    
    logger.info(f"Loading raw data from {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        
    try:
        data = np.load(raw_path)['data'] # shape (N, 15, bins, bins)
    except KeyError:
        # Fallback if 'data' key isn't used (though generate script uses 'data')
        with np.load(raw_path) as f:
            keys = list(f.keys())
            if len(keys) == 1:
                data = f[keys[0]]
            else:
                raise KeyError(f"Expected 'data' key or single key in npz. Found: {keys}")
                
    n_samples = data.shape[0]
    logger.info(f"Loaded {n_samples} samples with shape {data.shape[1:]}")
    
    # Shuffle and split
    logger.info(f"Splitting data with ratios {split_ratios} (seed={seed})")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    train_end = int(n_samples * split_ratios[0])
    val_end = train_end + int(n_samples * split_ratios[1])
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]
    
    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save as .npy instead of .npz for better multiprocessing performance
    train_path = os.path.join(processed_dir, config['dataset']['train_filename'])
    val_path = os.path.join(processed_dir, config['dataset']['val_filename'])
    test_path = os.path.join(processed_dir, config['dataset']['test_filename'])
    
    # Save as uncompressed .npy for memory-mapped access
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    np.save(test_path, test_data)
    
    logger.info(f"Saved processed datasets to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess frequency map data")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    preprocess_frequency_maps(args.config)

