import pickle
import matplotlib.pyplot as plt
import os

def plot_training_curves(history_path):
    """
    Plot training and validation loss curves from a pickle file.
    
    Supports plotting total loss and individual components (recon_loss, kl_loss)
    if available in the history file.
    """
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Extract data
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    
    # Check for auxiliary metrics
    train_metrics = history.get('train_metrics', {})
    val_metrics = history.get('val_metrics', {})
    
    has_metrics = bool(train_metrics) or bool(val_metrics)
    
    # Determine epochs
    max_epochs = max(len(train_losses), len(val_losses))
    if has_metrics and train_metrics:
        # Check lengths of metric arrays to be safe
        for v in train_metrics.values():
            max_epochs = max(max_epochs, len(v))
            
    epochs = range(1, max_epochs + 1)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # 1. Plot Total Loss
    if train_losses:
        plt.plot(epochs[:len(train_losses)], train_losses, label='Total Train Loss', 
                 linewidth=2, color='blue', linestyle='-')
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, label='Total Val Loss', 
                 linewidth=2, color='orange', linestyle='-')
        
    # 2. Plot Components (if available)
    colors = ['green', 'red', 'purple', 'brown']
    line_styles = ['--', ':']
    
    if has_metrics:
        # Get all metric keys from both train and val
        metric_keys = set()
        if train_metrics:
            metric_keys.update(train_metrics.keys())
        if val_metrics:
            metric_keys.update(val_metrics.keys())
            
        sorted_keys = sorted(list(metric_keys))
        
        for idx, key in enumerate(sorted_keys):
            color = colors[idx % len(colors)]
            
            # Train component
            if train_metrics and key in train_metrics:
                data = train_metrics[key]
                if data:
                    plt.plot(epochs[:len(data)], data, label=f'Train {key}', 
                             color=color, linestyle='--', alpha=0.7)
            
            # Val component
            if val_metrics and key in val_metrics:
                data = val_metrics[key]
                if data:
                    plt.plot(epochs[:len(data)], data, label=f'Val {key}', 
                             color=color, linestyle=':', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training Curves: {os.path.basename(history_path)}')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()
