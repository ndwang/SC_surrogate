import sys
import time
import yaml
import torch
from pathlib import Path
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from modeling.dataset import SpaceChargeDataset
from torch.utils.data import DataLoader
from modeling.models import create_model_from_config

# Settings
H5_PATH = 'data/processed/train.h5'
CONFIG_PATH = 'configs/training_config.yaml'
BATCH_SIZE = 1  # Match training config
NUM_BATCHES = 100  # Number of batches to benchmark
NUM_WORKERS = 0  # Use 0 for deterministic timing

def main():
    parser = argparse.ArgumentParser(description='Benchmark model forward pass time.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the benchmark on (cpu or cuda)')
    args = parser.parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        device = 'cpu'

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate model on selected device
    model = create_model_from_config(config)
    model.eval()
    model.to(device)

    # Instantiate dataset and dataloader
    dataset = SpaceChargeDataset(H5_PATH, device=device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    print(f'Benchmarking model forward pass on {device.upper()} for {NUM_BATCHES} batches (batch size={BATCH_SIZE})...')
    times = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= NUM_BATCHES:
                break
            start = time.time()
            _ = model(inputs)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f'Batch {i+1}: {elapsed:.4f}s')
    if times:
        avg = sum(times) / len(times)
        print(f'Average forward pass time per batch: {avg:.4f}s')
    else:
        print('No batches processed!')

if __name__ == '__main__':
    main() 