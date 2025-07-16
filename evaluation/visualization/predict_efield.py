import torch
import joblib
from modeling.dataset import SpaceChargeDataset
from modeling.models import create_model_from_config
import yaml
from .raw_data import plot_efield

def predict_and_visualize_efield(config_path, checkpoint_path, scalers_path, data_path, sample_idx=0):
    """Load model, scalers, and a sample; predict and visualize the denormalized electric field."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model_from_config(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Load scalers
    scalers = joblib.load(scalers_path)
    target_scaler = scalers['target_scaler']
    # Load sample
    dataset = SpaceChargeDataset(data_path, device='cpu')
    input_tensor, target_tensor = dataset[sample_idx]
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0]  # shape (3, Nx, Ny, Nz)
    # Denormalize
    pred_flat = pred.reshape(3, -1).T  # shape (N_voxels, 3)
    pred_denorm = target_scaler.inverse_transform(pred_flat).T.reshape(pred.shape)
    # Visualize
    plot_efield(pred_denorm.transpose(1,2,3,0), run_info=f"Predicted (sample {sample_idx})") 