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
│   └── dataset.py          # PyTorch Dataset & DataLoader utilities
├── evaluation/
│   └── visualize.py        # Interactive visualization tools
├── saved_models/           # Model checkpoints, scalers
├── tests/
│   └── test_data_pipeline.py    # Comprehensive test suite
├── environment.yml         # Conda environment definition
└── README.md
```

---

## 1. Data Generation

Generate synthetic space charge simulation data using Julia and Distgen:

```bash
python generation/generate_data.py configs/generation_config.yaml
```

- **Config:** `configs/generation_config.yaml` controls output location, grid size, number of samples, parameter ranges, and device (CPU/GPU).
- **Template:** Uses `configs/distgen_template.yaml` for beam/particle settings.
- **Output:** HDF5 file in `data/raw/` with group-per-sample structure: `run_00001/rho`, `run_00001/efield`, `run_00001/parameters`.

**Tip:** Requires Julia and the `SpaceCharge` Julia package. See [Julia/Distgen setup](#juliadistgen-setup) below if needed.

---

## 2. Data Preprocessing

Convert raw simulation data to a format suitable for PyTorch training:

```bash
python preprocessing/preprocess_data.py configs/training_config.yaml
```

Or in Python:

```python
from preprocessing.preprocess_data import Preprocessor
Preprocessor('configs/training_config.yaml').run()
```

**Pipeline steps:**
- Reads raw HDF5 data
- Converts to monolithic format for efficient loading
- Applies StandardScaler normalization
- Splits into train/val/test sets
- Saves processed data to `data/processed/` and scalers to `saved_models/`

---

## 3. Model Training & Dataset Usage

Load processed data for PyTorch training:

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

## 4. Evaluation & Visualization

Visualize charge density and electric field slices interactively:

```bash
python evaluation/visualize.py data/raw/simulations.h5 --plot both --run run_00000
```

- `--plot`: `density`, `efield`, or `both`
- `--run`: Specify the run/group to visualize (e.g., `run_00000`)

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
pytest tests/test_data_pipeline.py -v
```

- End-to-end pipeline validation
- Dataset/DataLoader integration
- Normalization correctness
- Error handling and edge cases

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
