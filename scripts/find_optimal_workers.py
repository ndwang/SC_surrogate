import sys
import os
import time
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from modeling.dataset import create_data_loaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def benchmark_workers(config_path, batch_size=None, num_batches=100, max_workers=None):
    config = load_config(config_path)
    
    # Override batch size if provided
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 32)
    
    # Get data path
    processed_dir = Path(config['paths']['processed_dir'])
    train_filename = config['dataset']['train_filename']
    train_path = str(processed_dir / train_filename)
    dataset_type = config.get('dataset', {}).get('type', 'space_charge')
    
    print(f"Benchmarking workers for dataset: {train_path}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches to measure: {num_batches}")
    
    # Determine worker counts to test
    if max_workers is None:
        max_workers = os.cpu_count()
        
    print(f"System CPU count: {os.cpu_count() or 'Unknown'}")
    print(f"Scanning up to {max_workers} workers...")

    # Always include 0 (main process)
    worker_counts = [0]
    
    # Add steps of 2 up to max_workers
    worker_counts.extend([i for i in range(2, max_workers + 1, 2)])
        
    # Remove duplicates and sort just in case
    worker_counts = sorted(list(set(worker_counts)))
    
    print(f"Worker counts to test: {worker_counts}")
    
    results = []
    
    for num_workers in worker_counts:
        print(f"\nTesting num_workers={num_workers}...")
        
        try:
            loader = create_data_loaders(
                data_path=train_path,
                dataset_type=dataset_type,
                batch_size=batch_size,
                num_workers=num_workers,
                device='cpu', # DataLoader loads to CPU memory usually
                shuffle=True,
                drop_last=True
            )
            
            # Warmup
            iterator = iter(loader)
            for _ in range(5):
                try:
                    next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    next(iterator)
            
            # Benchmark
            start_time = time.time()
            count = 0
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                count += 1
                
            end_time = time.time()
            duration = end_time - start_time
            throughput = (count * batch_size) / duration
            
            print(f"  Time: {duration:.4f}s")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            
            results.append({
                'num_workers': num_workers,
                'duration': duration,
                'throughput': throughput
            })
            
        except Exception as e:
            print(f"  Failed with error: {e}")
            
    # Find optimal
    if results:
        best_result = max(results, key=lambda x: x['throughput'])
        print("\n" + "="*40)
        print(f"Optimal num_workers: {best_result['num_workers']}")
        print(f"Max Throughput: {best_result['throughput']:.2f} samples/sec")
        print("="*40)
        
        # Also print full table
        print("\nFull Results:")
        print(f"{'Workers':<10} {'Time (s)':<10} {'Throughput (samples/s)':<25}")
        print("-" * 45)
        for r in results:
            print(f"{r['num_workers']:<10} {r['duration']:<10.4f} {r['throughput']:<25.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal number of workers for DataLoader')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size to use (default: from config)')
    parser.add_argument('--num-batches', type=int, default=50, help='Number of batches to measure')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of workers to test (default: CPU count)')
    
    args = parser.parse_args()
    
    benchmark_workers(args.config, args.batch_size, args.num_batches, args.max_workers)
