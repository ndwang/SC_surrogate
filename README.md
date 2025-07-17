# Space Charge Surrogate Model

A PyTorch-based surrogate model for space charge simulation, enabling rapid approximation of electric field calculations from charge density distributions. This repository provides a full pipeline: data generation (with Julia/Distgen), preprocessing, model training, evaluation, and visualization.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ndwang/SC_surrogate.git
cd SC_surrogate

# Create and activate the conda environment
conda env create -f environment.yml
conda activate sc_surrogate
```

---

## Project Structure

```
SC_surrogate/
├── configs/                # YAML configuration files
│   ├── training_config.yaml      # Main training & preprocessing config
│   ├── generation_config.yaml   # Data generation config
│   └── distgen_template.yaml    # Distgen beam template
├── data/
│   ├── raw/                # Raw simulation data (HDF5, group-per-sample)
│   └── processed/          # Preprocessed train/val/test data (HDF5, monolithic)
├── generation/
│   └── generate_data.py    # Data generation script (Julia/Distgen)
├── preprocessing/
│   └── preprocess_data.py  # Data preprocessing pipeline
├── modeling/
│   ├── models/             # Neural network model definitions
│   │   ├── __init__.py           # Model registry and factory
│   │   └── cnn3d.py              # 3D CNN architecture
│   ├── dataset.py          # PyTorch Dataset & DataLoader utilities
│   └── train.py            # Model training script
├── evaluation/
│   ├── evaluate.py         # Model evaluation script
│   └── visualize.py        # Interactive visualization tools
├── scripts/                # CLI entry points for main tasks and visualization
│   ├── generate_dataset.py       # Generate synthetic data
│   ├── preprocess_dataset.py     # Preprocess data
│   ├── train_model.py            # Train the model
│   ├── evaluate_model.py         # Evaluate the model
│   ├── visualize_raw_data.py     # Visualize raw data
│   ├── visualize_predict_efield.py # Visualize model predictions
│   └── visualize_training_curves.py # Visualize training/validation loss curves
├── saved_models/           # Model checkpoints, scalers
├── tests/
│   ├── test_data_pipeline.py    # Data pipeline test suite
│   └── test_model_training.py   # Model training test suite
├── environment.yml         # Conda environment definition
└── README.md
```

---

## 1. Data Generation

Generate synthetic space charge simulation data using Julia and Distgen:

```bash
python scripts/generate_dataset.py configs/generation_config.yaml
```

- **Config:** `configs/generation_config.yaml` controls output location, grid size, number of samples, parameter ranges, and device (CPU/GPU).
- **Template:** Uses `configs/distgen_template.yaml` for beam/particle settings.
- **Output:** HDF5 file in `data/raw/` with group-per-sample structure: `run_00001/rho`, `run_00001/efield`, `run_00001/parameters`.

**Tip:** Requires Julia and the `SpaceCharge` Julia package. See [Julia/Distgen setup](#juliadistgen-setup) below if needed.

---

## 2. Data Preprocessing

Convert raw simulation data to a format suitable for PyTorch training:

```bash
python scripts/preprocess_dataset.py --config configs/training_config.yaml
```

Or in Python:

```python
from preprocessing.preprocess_data import Preprocessor
Preprocessor('configs/training_config.yaml').run()
```

**Pipeline steps:**
- Reads raw HDF5 data
- Converts to monolithic format for efficient loading
- Applies normalization using configurable scalers (StandardScaler or SymlogScaler)
- Splits into train/val/test sets
- Saves processed data to `data/processed/` and scalers to `saved_models/`

**Scaler configuration:**
- You can specify the normalization method for both input (charge density) and target (electric field) data in `configs/training_config.yaml` as a dictionary with a `type` key and optional parameters:

```yaml
preprocessing:
  input_scaler:
    type: 'standard'   # Options: 'standard', 'symlog'
  target_scaler:
    type: 'symlog'     # Use 'symlog' for symmetric log scaling, or 'standard' for StandardScaler
    linthresh: 0.005   # (optional) Linear threshold for symlog
    percentile: 90     # (optional) Percentile for automatic linthresh selection
```
- `standard`: StandardScaler (mean=0, std=1, suitable for most data)
- `symlog`: SymlogScaler (handles data with both positive and negative values spanning orders of magnitude)
  - `linthresh`: (float, optional) Linear threshold for the symlog transform. If not provided, will be determined from data using `percentile`.
  - `percentile`: (float, optional) Percentile (0-100) of |x| to use for linthresh. Default is 90.

If not specified, both default to 'standard'.

---

## 3. Model Training

Train a neural network model on the preprocessed data:

```bash
python scripts/train_model.py --config configs/training_config.yaml
```

**Training Pipeline:**
- Automatically runs preprocessing if needed
- Creates model from config (currently supports CNN3D)
- Sets up data loaders, optimizer, scheduler, and loss function
- Includes validation, checkpointing, early stopping, and logging
- Saves best model, training history, and logs

**Loss Function Configuration:**
- Loss functions are now extensible and defined in `modeling/losses.py`.
- Standard losses: `mse`, `l1`/`mae`, `huber`.
- Custom/combined losses can be specified in the config as a dict, e.g.:

```yaml
training:
  loss_function:
    type: "combined"
    losses:
      - type: "mse"
      - type: "l1"
    weights: [0.7, 0.3]
```
- Add your own loss functions in `modeling/losses.py` and register them for use in config.

**Key Features:**
- **Model-agnostic:** Easily switch architectures via config
- **Reproducible:** Seed control and deterministic operations
- **Robust:** Automatic device selection, gradient clipping, error handling
- **Monitored:** Progress bars, logging, loss tracking

**Training Output:**
- `saved_models/best_model.pth` - Best model checkpoint
- `saved_models/latest_checkpoint.pth` - Latest model state
- `saved_models/training_history.pkl` - Loss curves and metrics
- `logs/training.log` - Detailed training logs

---

## 4. Model Evaluation

Evaluate a trained model on the test set:

```bash
python scripts/evaluate_model.py --config configs/training_config.yaml
```

Optional: specify a specific checkpoint
```bash
python scripts/evaluate_model.py --config configs/training_config.yaml --checkpoint saved_models/best_model.pth
```

**Evaluation Pipeline:**
- Automatically finds best model if no checkpoint specified
- Computes comprehensive metrics: MSE, MAE, R², RMSE
- Per-component metrics for each electric field component (Ex, Ey, Ez)
- Generates visualizations and saves predictions for analysis
- Creates human-readable reports

**Evaluation Output:**
- `saved_models/evaluation/evaluation_results.txt` - Summary report
- `saved_models/evaluation/evaluation_metrics.pkl` - Detailed metrics
- `saved_models/evaluation/predictions.npy` - Model predictions
- `saved_models/evaluation/plots/` - Visualization plots

---

## 5. Dataset Usage

Load processed data for custom PyTorch training:

```python
from modeling.dataset import SpaceChargeDataset, create_data_loaders

# Load a single dataset
dataset = SpaceChargeDataset('data/processed/train.h5')
input_tensor, target_tensor = dataset[0]  # input: (1,32,32,32), target: (3,32,32,32)

# Create DataLoaders for training/validation/testing
train_loader, val_loader, test_loader = create_data_loaders(
    'data/processed/train.h5',
    'data/processed/val.h5',
    'data/processed/test.h5',
    batch_size=8
)
```

---

## 6. Adding New Models

The framework supports easy addition of new model architectures:

### Step 1: Create Your Model
Create a new file `modeling/models/your_model.py`:

```python
import torch.nn as nn
from typing import Dict, Any

class YourModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config.get('model', {})
        # Initialize your model here
        
    def forward(self, x):
        # Your forward pass
        return output
        
    def get_model_summary(self):
        # Return model information
        return {'model_name': 'YourModel', ...}
```

### Step 2: Register Your Model
Edit `modeling/models/__init__.py`:

```python
from .your_model import YourModel

MODEL_REGISTRY = {
    'cnn3d': CNN3D,
    'your_model': YourModel,  # Add this line
}
```

### Step 3: Update Config
Set the architecture in `configs/training_config.yaml`:

```yaml
model:
  architecture: "your_model"
  # your model-specific parameters
```

### Step 4: Train & Evaluate
Use the same training and evaluation commands - the framework automatically uses your new model!

---

## 7. Visualization

The repository provides a collection of interactive visualization tools to help you explore raw data, training progress, and model predictions. All tools are located in `evaluation/visualization_tools/` and can be used both via command-line scripts and as Python modules.

### Visualization Tools Overview

- **Raw Data Visualization (`raw_data.py`)**
  - Visualize charge density, electric field, or both from raw HDF5 simulation files.
- **Model Prediction Visualization (`predict_efield.py`)**
  - Visualize model predictions versus ground truth, or inspect predicted fields for any test sample.
- **Training Curve Visualization (`training_curves.py`)**
  - Plot training and validation loss curves from saved training history.

### Example CLI Usage

**Visualize raw data (density, efield, or both):**
```bash
python scripts/visualize_raw_data.py data/raw/simulations.h5 --plot both --run run_00000
```
- `--plot`: Choose `density`, `efield`, or `both`
- `--run`: Specify the sample/run to visualize

**Visualize model predictions (compare or predict mode):**
```bash
python scripts/visualize_predict_efield.py data/processed/test.h5 --sample_idx 0 --checkpoint saved_models/best_model.pth --scalers saved_models/scalers.pkl --config configs/training_config.yaml --mode compare
```
- `--mode compare`: Interactive comparison of predicted and ground truth E-field
- `--mode predict`: Visualize charge density and predicted E-field only

**Plot training and validation loss curves:**
```bash
python scripts/visualize_training_curves.py saved_models/training_history.pkl
```

You can import and use all visualization functions directly in your Python scripts or notebooks:
```python
from evaluation.visualization_tools.raw_data import plot_density, plot_efield, plot_both
from evaluation.visualization_tools.predict_efield import plot_prediction_vs_truth
```

---

## Configuration

### Data Generation (`configs/generation_config.yaml`)
- `output_dir`, `output_filename`: Where to save raw data
- `template_file`: Path to distgen template
- `device`: `cpu` or `gpu`
- `grid_size`: Simulation grid resolution
- `n_samples`: Number of samples to generate
- `min_bound`, `max_bound`: Grid bounds (meters)
- `sigma_mins`, `sigma_maxs`: Parameter sampling ranges
- `seed`: Random seed for reproducibility

### Preprocessing & Training (`configs/training_config.yaml`)
- **Paths:** Raw/processed data, model save dir, logs
- **Preprocessing:** Split ratios, normalization, chunking
- **Model:** Architecture, channels, layers, activation, dropout, etc.
- **Training:** Batch size, epochs, optimizer, scheduler, loss, device
- **Evaluation:** Metrics, plotting, saving predictions
- **Logging:** Level, file, Tensorboard/W&B integration

### Distgen Template (`configs/distgen_template.yaml`)
- Defines the beam/particle distribution for simulation (see file for details)

---

## Data Format

### Raw Data (group-per-sample, HDF5)
- Each sample: `run_XXXXX/`
  - `rho`: Charge density, shape `(32, 32, 32)`, dtype `float64`
  - `efield`: Electric field, shape `(32, 32, 32, 3)`, dtype `float64`
  - `parameters`: Beam parameters, shape `(3,)`, dtype `float64`

### Processed Data (monolithic, HDF5)
- `charge_density`: shape `(N, 32, 32, 32)`, dtype `float32`, normalized
- `electric_field`: shape `(N, 3, 32, 32, 32)`, dtype `float32`, normalized
- For PyTorch: input `(1, Nx, Ny, Nz)`, target `(3, Nx, Ny, Nz)`

---

## Testing & Quality Assurance

Run the full test suite:

```bash
# Test data pipeline
pytest tests/test_data_pipeline.py -v

# Test model training pipeline
pytest tests/test_model_training.py -v

# Run all tests
pytest tests/ -v
```

**Data Pipeline Tests:**
- End-to-end pipeline validation
- Dataset/DataLoader integration
- Normalization correctness
- Error handling and edge cases

**Model Training Tests:**
- Model instantiation and forward pass
- Training loop functionality
- Checkpoint saving/loading
- Evaluation pipeline
- End-to-end training process

**Linting & Type Checking:**
```bash
ruff check --fix
mypy preprocessing/ modeling/ --ignore-missing-imports
```

---

## Julia Setup (for Data Generation)

- Install [Julia](https://julialang.org/downloads/)
- Install Julia packages:
  - `SpaceCharge` (for field calculation)
  - `CUDA` (if using GPU)
- The Python package `juliacall` (installed via pip) is used for Python-Julia interop
