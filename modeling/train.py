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

from modeling.models import create_model_from_config  # noqa: E402
from modeling.dataset import create_data_loaders  # noqa: E402
from preprocessing.preprocess_data import Preprocessor  # noqa: E402
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
    
    def _ensure_data_processed(self) -> None:
        """Ensure preprocessed data exists, run preprocessing if needed."""
        processed_dir = Path(self.config['paths']['processed_dir'])
        
        # Check if processed data exists
        train_file = processed_dir / 'train.h5'
        val_file = processed_dir / 'val.h5'
        test_file = processed_dir / 'test.h5'
        
        if not (train_file.exists() and val_file.exists() and test_file.exists()):
            self.logger.info("Processed data not found, running preprocessing...")
            preprocessor = Preprocessor(self.config_path)
            preprocessor.run()
            self.logger.info("Preprocessing completed")
        else:
            self.logger.info("Using existing processed data")
    
    def setup_data(self) -> None:
        """Setup data loaders."""
        self._ensure_data_processed()
        
        # Get paths
        processed_dir = Path(self.config['paths']['processed_dir'])
        train_path = str(processed_dir / 'train.h5')
        val_path = str(processed_dir / 'val.h5')
        test_path = str(processed_dir / 'test.h5')
        
        # Get training config
        training_config = self.config.get('training', {})
        batch_size = int(training_config.get('batch_size', 8))
        num_workers = int(training_config.get('num_workers', 4))
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device='cpu'  # Move to GPU in training loop
        )
        
        self.logger.info("Data loaders created:")
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
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1}",
            leave=False
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
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
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
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
        
        # Save regular checkpoint
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        
        # Save latest checkpoint
        latest_path = save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        early_stop_config = self.config.get('training', {}).get('early_stopping', {})
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
        
        # Add outer progress bar for epochs
        epoch_progress_bar = tqdm(range(num_epochs), desc="Epochs", position=0)
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
                
                # Save checkpoint
                if epoch % save_frequency == 0 or is_best:
                    self.save_checkpoint(epoch, val_loss, is_best)
                
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
        
        # Final save
        if len(self.val_losses) > 0:
            final_val_loss = self.val_losses[-1]
        else:
            final_val_loss = self.validate(criterion)
        
        self.save_checkpoint(self.current_epoch, final_val_loss)
        
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