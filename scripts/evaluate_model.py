import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))
from evaluation.evaluate import Evaluator

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Space Charge Surrogate Model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (if not provided, will use best model)')
    args = parser.parse_args()
    evaluator = Evaluator(args.config, args.checkpoint)
    evaluator.evaluate()

if __name__ == '__main__':
    main() 