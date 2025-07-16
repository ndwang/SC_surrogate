import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))
from preprocessing.preprocess_data import Preprocessor

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess space charge data.')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for preprocessing')
    args = parser.parse_args()
    preprocessor = Preprocessor(args.config, batch_size=args.batch_size)
    preprocessor.run()

if __name__ == '__main__':
    main() 