"""
Model-agnostic training script for Space Charge Surrogate Model.

This script loads configuration, creates the model via the registry, sets up
data loaders, and runs the training loop with validation, checkpointing,
early stopping, and logging.
"""

import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
from typing import Dict, Any
import time
from tqdm import tqdm
import pickle

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from modeling.models import create_model_from_config
from modeling.dataset import create_data_loaders, DATASET_REGISTRY
from modeling.loss import get_loss_from_config


class Trainer:
    """
    Trainer class that handles the complete training pipeline.
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        self.logger.info("Trainer initialized successfully")
    
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
        
        # Close data loaders if they exist
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            if hasattr(self.train_loader.dataset, 'close'):
                try:
                    self.train_loader.dataset.close()
                except Exception:
                    pass
        
        if hasattr(self, 'val_loader') and self.val_loader is not None:
            if hasattr(self.val_loader.dataset, 'close'):
                try:
                    self.val_loader.dataset.close()
                except Exception:
                    pass
        
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
        logger = logging.getLogger('space_charge_trainer')
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
        
        # File handler if enabled
        if log_config.get('log_to_file', False):
            log_dir = Path(self.config['paths']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / log_config.get('log_filename', 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def setup_data(self) -> None:
        """Setup data loaders."""        
        # Get dataset type from config
        dataset_config = self.config.get('dataset', {})
        dataset_type = dataset_config.get('type', 'space_charge')
        
        # Validate dataset type
        if dataset_type not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Supported types: {list(DATASET_REGISTRY.keys())}"
            )
        
        # Get filenames from config (required)
        train_filename = dataset_config.get('train_filename')
        val_filename = dataset_config.get('val_filename')
        test_filename = dataset_config.get('test_filename')
        
        if not all([train_filename, val_filename, test_filename]):
            raise ValueError(
                "All dataset filenames must be specified: "
                "train_filename, val_filename, test_filename"
            )
        
        # Build paths using processed_dir + filenames
        processed_dir = Path(self.config['paths']['processed_dir'])
        train_path = str(processed_dir / train_filename)
        val_path = str(processed_dir / val_filename)
        test_path = str(processed_dir / test_filename)
        
        # Get training config
        training_config = self.config.get('training', {})
        batch_size = int(training_config.get('batch_size', 8))
        num_workers = int(training_config.get('num_workers', 4))
        
        # Create data loaders
        self.train_loader = create_data_loaders(
            data_path=train_path,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            device='cpu',  # Move to GPU in training loop
            shuffle=True
        )
        self.val_loader = create_data_loaders(
            data_path=val_path,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            device='cpu',
            shuffle=False
        )
        self.test_loader = create_data_loaders(
            data_path=test_path,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            device='cpu',
            shuffle=False
        )
        
        self.logger.info(f"Data loaders created for dataset type: {dataset_type}")
        self.logger.info(f"  Train: {len(self.train_loader)} batches")
        self.logger.info(f"  Validation: {len(self.val_loader)} batches")
        self.logger.info(f"  Test: {len(self.test_loader)} batches")
    
    def setup_model(self) -> None:
        """Setup model and move to device."""
        self.model = create_model_from_config(self.config)
        self.model.to(self.device)
        
        # Log model info
        model_summary = self.model.get_model_summary()
        self.logger.info(f"Model created: {model_summary['model_name']}")
        self.logger.info(f"Parameters: {model_summary['total_parameters']:,}")
        self.logger.info(f"Device: {self.device}")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config.get('training', {})
        
        # Setup optimizer
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        learning_rate = float(training_config.get('learning_rate', 0.001))
        weight_decay = float(training_config.get('weight_decay', 1e-5))
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'plateau').lower()
        factor = float(scheduler_config.get('factor', 0.5))
        patience = int(scheduler_config.get('patience', 10))
        min_lr = float(scheduler_config.get('min_lr', 1e-6))
        step_size = int(scheduler_config.get('step_size', 30))
        gamma = float(scheduler_config.get('gamma', 0.1))
        num_epochs = int(training_config.get('num_epochs', 100))
        
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {optimizer_type}, LR: {learning_rate}")
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        loss_config = self.config.get('training', {}).get('loss_function', 'mse')
        return get_loss_from_config(loss_config)
    
    def train_epoch(self, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Get dataset type to handle different data formats
        dataset_type = self.config.get('dataset', {}).get('type', 'space_charge')
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1}",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle different dataset return types
            if dataset_type == 'frequency_map':
                # FrequencyMapDataset returns a single tensor (for VAE/reconstruction)
                # Use the same data as both input and target
                inputs = batch_data.to(self.device, non_blocking=True)
                targets = inputs  # For reconstruction tasks
            else:
                # SpaceChargeDataset returns (input, target) tuple
                inputs, targets = batch_data
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.6f}'
            })
        
        return total_loss / num_batches
    
    def validate(self, criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Get dataset type to handle different data formats
        dataset_type = self.config.get('dataset', {}).get('type', 'space_charge')
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle different dataset return types
                if dataset_type == 'frequency_map':
                    # FrequencyMapDataset returns a single tensor (for VAE/reconstruction)
                    # Use the same data as both input and target
                    inputs = batch_data.to(self.device, non_blocking=True)
                    targets = inputs  # For reconstruction tasks
                else:
                    # SpaceChargeDataset returns (input, target) tuple
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint.

        When is_best is True, only the best model artifact is written.
        Regular checkpoints are written only when explicitly requested
        by the training loop (e.g., at save_frequency).
        """
        save_dir = Path(self.config['paths']['model_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save regular checkpoint only when not saving best-only
        if not is_best:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {val_loss:.6f}")

    def maybe_resume_from_checkpoint(self) -> None:
        """Resume training state from a checkpoint if enabled in config."""
        training_config = self.config.get('training', {})
        resume_cfg = training_config.get('resume', {}) or {}
        enabled = bool(resume_cfg.get('enabled', False))
        if not enabled:
            return

        save_dir = Path(self.config['paths']['model_save_dir'])
        use_best = bool(resume_cfg.get('use_best', False))
        checkpoint_path_cfg = resume_cfg.get('checkpoint_path')

        if use_best:
            checkpoint_path = save_dir / 'best_model.pth'
        elif checkpoint_path_cfg:
            checkpoint_path = Path(str(checkpoint_path_cfg))
        else:
            checkpoint_path = save_dir / 'latest_checkpoint.pth'

        if not checkpoint_path.exists():
            self.logger.warning(f"Resume enabled but checkpoint not found at: {checkpoint_path}")
            return

        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model/optimizer/scheduler
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                self.logger.warning("Optimizer state could not be loaded; continuing with fresh optimizer")
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                self.logger.warning("Scheduler state could not be loaded; continuing with fresh scheduler")

        # Restore history and counters
        self.current_epoch = int(checkpoint.get('epoch', -1)) + 1
        self.train_losses = checkpoint.get('train_losses', []) or []
        self.val_losses = checkpoint.get('val_losses', []) or []
        self.best_val_loss = float(checkpoint.get('val_loss', self.best_val_loss))
        self.logger.info(f"Resumed at epoch index {self.current_epoch} (next epoch = {self.current_epoch+1})")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        early_stop_config = self.config.get('training', {}).get('early_stopping', {})
        # Allow disabling early stopping via config: training.early_stopping.enabled: false
        enabled = bool(early_stop_config.get('enabled', True))
        if not enabled:
            return False
        patience = int(early_stop_config.get('patience', 20))
        min_delta = float(early_stop_config.get('min_delta', 1e-6))
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def train(self) -> None:
        """Main training loop."""
        # Setup components
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        criterion = self.setup_loss_function()
        
        # Training configuration
        training_config = self.config.get('training', {})
        num_epochs = int(training_config.get('num_epochs', 100))
        validation_frequency = int(training_config.get('validation_frequency', 1))
        save_frequency = int(training_config.get('save_frequency', 10))
        
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Validation frequency: {validation_frequency}")
        
        start_time = time.time()
        
        # Optionally resume from checkpoint
        self.maybe_resume_from_checkpoint()

        # Add outer progress bar for epochs (respect resume start)
        epoch_progress_bar = tqdm(range(self.current_epoch, num_epochs), desc="Epochs", position=0)
        for epoch in epoch_progress_bar:
            self.current_epoch = epoch
            # Update outer bar description
            epoch_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % validation_frequency == 0:
                val_loss = self.validate(criterion)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                
                # Save regular checkpoint only at configured frequency
                if epoch % save_frequency == 0:
                    self.save_checkpoint(epoch, val_loss, is_best=False)

                # Save best model separately without writing a regular checkpoint
                if is_best:
                    # Update tracking so subsequent comparisons/logging are correct
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                
                # Log progress
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e}"
                )
                
                # Early stopping
                if self.check_early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # No validation this epoch
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        # No unconditional final regular checkpoint; regular checkpoints are only
        # saved at configured frequency. Best model has already been saved when achieved.
        
        # Training summary
        total_time = time.time() - start_time
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save training history
        history_path = Path(self.config['paths']['model_save_dir']) / 'training_history.pkl'
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time
        }
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        self.logger.info(f"Training history saved to {history_path}") 