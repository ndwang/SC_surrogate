import time
from modeling.dataset import SpaceChargeDataset
from torch.utils.data import DataLoader

# Settings
H5_PATH = 'data/processed/train.h5'
BATCH_SIZE = 16  # You can adjust this as needed
WORKER_OPTIONS = [0, 2, 4, 8, 12, 16, 20, 24]

def main():
    # Instantiate dataset (on CPU for benchmarking)
    dataset = SpaceChargeDataset(H5_PATH, device='cpu')

    print('Benchmarking DataLoader with different num_workers...')
    for num_workers in WORKER_OPTIONS:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= 10:
                break
        elapsed = time.time() - start
        print(f'num_workers={num_workers}: {elapsed:.2f}s for {BATCH_SIZE} batches')

if __name__ == '__main__':
    main() 