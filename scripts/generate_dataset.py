import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))
from generation.generate_data import SpaceChargeDataGenerator

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate space charge data.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with SpaceChargeDataGenerator.from_config(args.config) as generator:
        generator.run()

if __name__ == '__main__':
    main() 