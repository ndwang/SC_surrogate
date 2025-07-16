import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from evaluation.visualization.training_curves import plot_training_curves

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss curves from training_history.pkl.')
    parser.add_argument('history_file', type=str, help='Path to training_history.pkl')
    args = parser.parse_args()

    if not os.path.isfile(args.history_file):
        print(f"Error: File '{args.history_file}' not found.")
        sys.exit(1)
    plot_training_curves(args.history_file)

if __name__ == '__main__':
    main() 