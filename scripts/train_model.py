import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))
from modeling.train import Trainer

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Space Charge Surrogate Model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Path to training configuration file')
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main() 