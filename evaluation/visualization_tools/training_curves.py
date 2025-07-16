import pickle
import matplotlib.pyplot as plt

def plot_training_curves(history_path):
    """Plot training and validation loss curves from a pickle file."""
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    epochs = range(1, max(len(train_losses), len(val_losses)) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss')
    plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() 