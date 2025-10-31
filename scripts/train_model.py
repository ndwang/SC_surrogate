import sys
import shutil
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
    
    # Copy the used config file into the model save directory before training
    save_dir = Path(trainer.config['paths']['model_save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    dst_path = save_dir / 'training_config.yaml'
    shutil.copy2(args.config, dst_path)

    trainer.train()

if __name__ == '__main__':
    main() 