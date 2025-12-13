import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from modeling.models import create_model_from_config
from modeling.dataset import create_data_loaders

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def benchmark_batch_size(config_path, num_workers=4):
    config = load_config(config_path)
    
    # Device setup
    device_name = config.get('training', {}).get('device', 'auto')
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"Benchmarking batch sizes on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # Get data path
    processed_dir = Path(config['paths']['processed_dir'])
    train_filename = config['dataset']['train_filename']
    train_path = str(processed_dir / train_filename)
    dataset_type = config.get('dataset', {}).get('type', 'space_charge')
    
    # Create model once to get structure (will re-instantiate or deepcopy if needed, 
    # but actually we just need one instance if we are just testing throughput/memory.
    # However, if batch size changes cause reallocation issues, we should be careful.
    # For now, one model instance is fine.)
    model = create_model_from_config(config)
    model.to(device)
    model.train()
    
    # Setup simple optimizer/loss for realistic memory usage
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Batch sizes to test: powers of 2
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    results = []
    
    for bs in batch_sizes:
        print(f"\nTesting batch_size={bs}...")
        
        try:
            # Create loader with this batch size
            loader = create_data_loaders(
                data_path=train_path,
                dataset_type=dataset_type,
                batch_size=bs,
                num_workers=num_workers,
                device='cpu',
                shuffle=True,
                drop_last=True
            )
            
            if len(loader) == 0:
                print("  Skipping: Dataset too small for this batch size.")
                continue
                
            # Run a few iterations
            # We want to measure throughput (samples/sec) and check for OOM
            
            total_samples = 0
            start_time = time.time()
            
            # Use a limit on number of batches to keep benchmark fast
            # but enough to stabilize
            limit_batches = 20
            
            for i, batch_data in enumerate(loader):
                if i >= limit_batches:
                    break
                
                # Handle data
                if dataset_type == 'frequency_map':
                    inputs = batch_data.to(device)
                    targets = inputs
                else:
                    inputs, targets = batch_data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                
                # Optimization step
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Handle tuple output from VAE (recon, mu, logvar)
                if isinstance(outputs, tuple):
                    # For benchmarking throughput, we can just use the reconstruction part 
                    # for the loss calculation if we are using simple MSELoss
                    recon = outputs[0]
                    loss = criterion(recon, targets)
                else:
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                total_samples += bs
            
            end_time = time.time()
            
            if total_samples == 0:
                 print("  No samples processed.")
                 continue

            duration = end_time - start_time
            throughput = total_samples / duration
            
            print(f"  Success.")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            
            # Record GPU memory if possible
            mem_usage = "N/A"
            if device.type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_usage = f"{mem_alloc:.2f} GB (Alloc), {mem_reserved:.2f} GB (Res)"
                print(f"  Memory: {mem_usage}")
                
            results.append({
                'batch_size': bs,
                'throughput': throughput,
                'memory': mem_usage
            })
            
            # Check for significant throughput drop (indicating swapping/thrashing)
            if len(results) > 1:
                # Get max throughput from *previous* results
                max_throughput_so_far = max(r['throughput'] for r in results)
                
                # If current is significantly worse than best, stop.
                # A small drop might just be noise or slight inefficiency, but swapping causes massive drops.
                # We'll use a threshold of 60% of max throughput.
                if throughput < max_throughput_so_far * 0.6:
                    print(f"  Warning: Significant throughput drop detected ({throughput:.2f} < 0.6 * {max_throughput_so_far:.2f}).")
                    print(f"  This likely indicates GPU memory exhaustion and swapping to system RAM.")
                    print(f"  Stopping benchmark.")
                    break

            # Clear cache to be fair to next size
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM: Out of Memory at batch_size={bs}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break # Stop testing larger sizes
            else:
                print(f"  Failed with error: {e}")
                break
        except Exception as e:
            print(f"  Failed with error: {e}")
            break
            
    # Summary
    if results:
        best_result = max(results, key=lambda x: x['throughput'])
        print("\n" + "="*40)
        print(f"Highest Throughput Batch Size: {best_result['batch_size']}")
        print(f"Max Throughput: {best_result['throughput']:.2f} samples/sec")
        print("="*40)
        
        print("\nFull Results:")
        print(f"{'Batch Size':<12} {'Throughput (samples/s)':<25} {'Memory':<20}")
        print("-" * 60)
        for r in results:
            print(f"{r['batch_size']:<12} {r['throughput']:<25.2f} {r['memory']:<20}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal batch size for training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to use for data loading during benchmark')
    
    args = parser.parse_args()
    
    benchmark_batch_size(args.config, args.num_workers)
