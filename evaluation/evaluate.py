"""
Model-agnostic evaluation script for Space Charge Surrogate Model.

This script loads a trained model, evaluates it on the test set, and computes
comprehensive metrics including MSE, MAE, R², and optional visualizations.
"""

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pickle

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from modeling.models import create_model_from_config  # noqa: E402
from modeling.dataset import create_data_loaders  # noqa: E402


class Evaluator:
    """
    Evaluator class for model evaluation and metrics computation.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to training configuration file
            checkpoint_path: Optional specific checkpoint path. If None, auto-detects best checkpoint.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Initialize state
        self.model = None
        self.test_loader = None
        
        # Find checkpoint
        if checkpoint_path is None:
            self.checkpoint_path = self._find_best_checkpoint()
        else:
            self.checkpoint_path = checkpoint_path
            
        self.logger.info("Evaluator initialized successfully")
    
    def cleanup(self):
        """
        Clean up resources to prevent file locking issues on Windows.
        Closes logging handlers and releases file handles.
        """
        # Close all handlers for this logger
        for handler in self.logger.handlers[:]:
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass
        
        # Close test loader if it exists
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            if hasattr(self.test_loader.dataset, 'close'):
                try:
                    self.test_loader.dataset.close()
                except Exception:
                    pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = self.config.get('training', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logger
        logger = logging.getLogger('space_charge_evaluator')
        logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _find_best_checkpoint(self) -> str:
        """Find the best model checkpoint."""
        model_dir = Path(self.config['paths']['model_save_dir'])
        
        # Look for best model first
        best_model_path = model_dir / 'best_model.pth'
        if best_model_path.exists():
            self.logger.info(f"Using best model: {best_model_path}")
            return str(best_model_path)
        
        # Fallback to latest checkpoint
        latest_checkpoint_path = model_dir / 'latest_checkpoint.pth'
        if latest_checkpoint_path.exists():
            self.logger.info(f"Using latest checkpoint: {latest_checkpoint_path}")
            return str(latest_checkpoint_path)
        
        # Look for any checkpoint files
        checkpoint_files = list(model_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoint_files:
            # Get the most recent one
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Using checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)
        
        raise FileNotFoundError(f"No model checkpoints found in {model_dir}")
    
    def load_model(self) -> None:
        """Load the trained model from checkpoint."""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create model from config (either from checkpoint or current config)
        model_config = checkpoint.get('config', self.config)
        self.model = create_model_from_config(model_config)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        model_summary = self.model.get_model_summary()
        self.logger.info(f"Model loaded: {model_summary['model_name']}")
        self.logger.info(f"Parameters: {model_summary['total_parameters']:,}")
        self.logger.info(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
    
    def setup_data(self) -> None:
        """Setup data loader for evaluation."""
        # Get paths
        processed_dir = Path(self.config['paths']['processed_dir'])
        train_path = str(processed_dir / 'train.h5')
        val_path = str(processed_dir / 'val.h5')
        test_path = str(processed_dir / 'test.h5')
        
        # Get training config
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 8)
        num_workers = training_config.get('num_workers', 4)
        
        # Create data loaders
        _, _, self.test_loader = create_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device='cpu'  # Move to GPU during evaluation
        )
        
        self.logger.info(f"Test loader created: {len(self.test_loader)} batches")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def predict_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on all test data and return predictions and targets."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        self.logger.info("Running inference on test set...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                
                # Move to CPU and convert to numpy
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        self.logger.info(f"Inference completed. Shape: {predictions.shape}")
        return predictions, targets
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        self.logger.info("Computing evaluation metrics...")
        
        # Get evaluation config
        eval_config = self.config.get('evaluation', {})
        metrics_to_compute = eval_config.get('metrics', ['mse', 'mae', 'r2_score'])
        
        metrics = {}
        
        # Flatten arrays for overall metrics
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # Overall metrics
        if 'mse' in metrics_to_compute:
            mse = mean_squared_error(target_flat, pred_flat)
            metrics['mse'] = float(mse)
        
        if 'mae' in metrics_to_compute:
            mae = mean_absolute_error(target_flat, pred_flat)
            metrics['mae'] = float(mae)
        
        if 'r2_score' in metrics_to_compute:
            r2 = r2_score(target_flat, pred_flat)
            metrics['r2_score'] = float(r2)
        
        # RMSE (always compute as it's commonly used)
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        metrics['rmse'] = float(rmse)
        
        # Per-component metrics (for electric field components)
        component_metrics = {}
        num_components = predictions.shape[1]  # Should be 3 for Ex, Ey, Ez
        component_names = ['Ex', 'Ey', 'Ez'] if num_components == 3 else [f'Component_{i}' for i in range(num_components)]
        
        for i, comp_name in enumerate(component_names):
            pred_comp = predictions[:, i].flatten()
            target_comp = targets[:, i].flatten()
            
            comp_metrics = {}
            if 'mse' in metrics_to_compute:
                comp_metrics['mse'] = float(mean_squared_error(target_comp, pred_comp))
            if 'mae' in metrics_to_compute:
                comp_metrics['mae'] = float(mean_absolute_error(target_comp, pred_comp))
            if 'r2_score' in metrics_to_compute:
                comp_metrics['r2_score'] = float(r2_score(target_comp, pred_comp))
            comp_metrics['rmse'] = float(np.sqrt(mean_squared_error(target_comp, pred_comp)))
            
            component_metrics[comp_name] = comp_metrics
        
        metrics['per_component'] = component_metrics
        
        # Additional statistics
        stats = {
            'prediction_stats': {
                'mean': float(pred_flat.mean()),
                'std': float(pred_flat.std()),
                'min': float(pred_flat.min()),
                'max': float(pred_flat.max())
            },
            'target_stats': {
                'mean': float(target_flat.mean()),
                'std': float(target_flat.std()),
                'min': float(target_flat.min()),
                'max': float(target_flat.max())
            },
            'relative_error': {
                'mean': float(np.mean(np.abs((pred_flat - target_flat) / (target_flat + 1e-8)))),
                'std': float(np.std(np.abs((pred_flat - target_flat) / (target_flat + 1e-8))))
            }
        }
        
        metrics['statistics'] = stats
        
        return metrics
    
    def save_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Save predictions and targets for later analysis."""
        eval_config = self.config.get('evaluation', {})
        
        if eval_config.get('save_predictions', True):
            save_dir = Path(self.config['paths']['model_save_dir']) / 'evaluation'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays
            np.save(save_dir / 'predictions.npy', predictions)
            np.save(save_dir / 'targets.npy', targets)
            
            # Save metadata
            metadata = {
                'shape': predictions.shape,
                'num_samples': predictions.shape[0],
                'num_components': predictions.shape[1],
                'checkpoint_path': self.checkpoint_path,
                'config_path': self.config_path
            }
            
            with open(save_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.info(f"Predictions saved to {save_dir}")
    
    def create_visualizations(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Create visualization plots."""
        eval_config = self.config.get('evaluation', {})
        max_plots = eval_config.get('max_plots', 5)
        
        save_dir = Path(self.config['paths']['model_save_dir']) / 'evaluation' / 'plots'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prediction vs Target scatter plots
        num_components = predictions.shape[1]
        component_names = ['Ex', 'Ey', 'Ez'] if num_components == 3 else [f'Component_{i}' for i in range(num_components)]
        
        fig, axes = plt.subplots(1, num_components, figsize=(5*num_components, 4))
        if num_components == 1:
            axes = [axes]
        
        for i, (ax, comp_name) in enumerate(zip(axes, component_names)):
            pred_comp = predictions[:, i].flatten()
            target_comp = targets[:, i].flatten()
            
            # Sample points for plotting (to avoid overcrowding)
            max_points = 10000
            if len(pred_comp) > max_points:
                indices = np.random.choice(len(pred_comp), max_points, replace=False)
                pred_comp = pred_comp[indices]
                target_comp = target_comp[indices]
            
            ax.scatter(target_comp, pred_comp, alpha=0.5, s=1)
            ax.plot([target_comp.min(), target_comp.max()], 
                   [target_comp.min(), target_comp.max()], 'r--', linewidth=2)
            ax.set_xlabel(f'True {comp_name}')
            ax.set_ylabel(f'Predicted {comp_name}')
            ax.set_title(f'{comp_name} Prediction vs Truth')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_vs_truth.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution plots
        fig, axes = plt.subplots(1, num_components, figsize=(5*num_components, 4))
        if num_components == 1:
            axes = [axes]
        
        for i, (ax, comp_name) in enumerate(zip(axes, component_names)):
            pred_comp = predictions[:, i].flatten()
            target_comp = targets[:, i].flatten()
            errors = pred_comp - target_comp
            
            ax.hist(errors, bins=50, alpha=0.7, density=True)
            ax.axvline(errors.mean(), color='red', linestyle='--', 
                      label=f'Mean: {errors.mean():.2e}')
            ax.set_xlabel(f'Prediction Error ({comp_name})')
            ax.set_ylabel('Density')
            ax.set_title(f'{comp_name} Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sample field visualizations (2D slices)
        if predictions.ndim == 5:  # (N, C, Nx, Ny, Nz)
            n_samples_to_plot = min(max_plots, predictions.shape[0])
            
            for sample_idx in range(n_samples_to_plot):
                fig, axes = plt.subplots(2, num_components, figsize=(5*num_components, 8))
                if num_components == 1:
                    axes = axes.reshape(-1, 1)
                
                # Take middle slice in Z direction
                z_slice = predictions.shape[-1] // 2
                
                for comp_idx, comp_name in enumerate(component_names):
                    # Predicted field
                    pred_slice = predictions[sample_idx, comp_idx, :, :, z_slice]
                    im1 = axes[0, comp_idx].imshow(pred_slice, cmap='RdBu_r')
                    axes[0, comp_idx].set_title(f'Predicted {comp_name} (Sample {sample_idx})')
                    plt.colorbar(im1, ax=axes[0, comp_idx])
                    
                    # True field
                    true_slice = targets[sample_idx, comp_idx, :, :, z_slice]
                    im2 = axes[1, comp_idx].imshow(true_slice, cmap='RdBu_r')
                    axes[1, comp_idx].set_title(f'True {comp_name} (Sample {sample_idx})')
                    plt.colorbar(im2, ax=axes[1, comp_idx])
                
                plt.tight_layout()
                plt.savefig(save_dir / f'field_comparison_sample_{sample_idx:03d}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        self.logger.info(f"Visualizations saved to {save_dir}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        # Setup components
        self.load_model()
        self.setup_data()
        
        # Run inference
        predictions, targets = self.predict_all()
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)
        
        # Save results
        eval_config = self.config.get('evaluation', {})
        if eval_config.get('save_predictions', True):
            self.save_predictions(predictions, targets)
        
        # Create visualizations
        self.create_visualizations(predictions, targets)
        
        # Save metrics
        save_dir = Path(self.config['paths']['model_save_dir']) / 'evaluation'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'evaluation_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        
        # Save human-readable metrics
        with open(save_dir / 'evaluation_results.txt', 'w') as f:
            f.write("Space Charge Surrogate Model - Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Test samples: {predictions.shape[0]}\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric_name, value in metrics.items():
                if metric_name not in ['per_component', 'statistics']:
                    f.write(f"{metric_name.upper()}: {value:.6e}\n")
            
            f.write("\nPer-Component Metrics:\n")
            f.write("-" * 25 + "\n")
            for comp_name, comp_metrics in metrics['per_component'].items():
                f.write(f"\n{comp_name}:\n")
                for metric_name, value in comp_metrics.items():
                    f.write(f"  {metric_name.upper()}: {value:.6e}\n")
        
        # Log summary
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Overall MSE: {metrics.get('mse', 'N/A'):.6e}")
        self.logger.info(f"Overall MAE: {metrics.get('mae', 'N/A'):.6e}")
        self.logger.info(f"Overall R²: {metrics.get('r2_score', 'N/A'):.6f}")
        self.logger.info(f"Results saved to {save_dir}")
        
        return metrics


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Space Charge Surrogate Model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (if not provided, will use best model)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = Evaluator(args.config, args.checkpoint)
    metrics = evaluator.evaluate()
    
    return metrics


if __name__ == "__main__":
    main() 