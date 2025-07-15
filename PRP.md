### name: "Implement Data Preprocessing and PyTorch Dataset for Surrogate Model"
Create the foundational data pipeline components for the space charge surrogate model. This involves processing raw simulation data (charge density grids and corresponding electric field grids) into a model-ready format and creating an efficient PyTorch Dataset class for loading this data during training.

## Goal
To build a robust and reproducible data preprocessing script and a PyTorch `Dataset` class. The script will read the raw HDF5 file (which contains one group per simulation run), convert it into a more efficient monolithic format, apply normalization, and split it into train/validation/test sets. The Dataset class will then provide an efficient interface to load these processed files for model training.

## Why
-   **Model Readiness**: Raw simulation data is not in an optimal format for training neural networks. We need to normalize it for stable training and split it correctly for unbiased evaluation.
-   **Efficiency**: Loading large datasets from disk can be a bottleneck. A custom PyTorch `Dataset` that reads from HDF5 files allows for lazy loading, which is memory-efficient. The raw group-per-sample HDF5 format is inefficient for batch loading. Converting to large, contiguous datasets in the processed files will significantly speed up training I/O.
-   **Reproducibility**: A scripted preprocessing pipeline ensures that every experiment uses data prepared in the exact same way.
-   **Foundation**: This is a critical prerequisite for building the model training (`train.py`) and evaluation (`evaluate.py`) scripts.

## What
The implementation will deliver two core Python files and one configuration file.

### Success Criteria
-   [ ] A new `configs/training_config.yaml` file exists with sections for paths, preprocessing, model, and training hyperparameters.
-   [ ] A new `preprocessing/preprocess_data.py` script is created.
-   [ ] Running `preprocess_data.py` successfully reads `data/raw/space_charge_data.h5` and generates `train.h5`, `val.h5`, and `test.h5` in `data/processed/`.
-   [ ] The script also generates `saved_models/scalers.pkl` containing the fitted normalization scalers.
-   [ ] Data within the processed HDF5 files is verified to be normalized. For StandardScaler, the mean should be approximately 0 and the standard deviation approximately 1.
-   [ ] A new `modeling/dataset.py` file is created containing a `SpaceChargeDataset` class inheriting from `torch.utils.data.Dataset`.
-   [ ] The `SpaceChargeDataset` can be instantiated with a path to a processed `.h5` file.
-   [ ] The dataset, when wrapped in a `torch.utils.data.DataLoader`, successfully yields batches of input and target tensors of the correct shape and `dtype` (`torch.float32`).

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  why: Official PyTorch guide on creating custom Datasets and DataLoaders. The pattern shown here is exactly what we need.

- url: https://h5py.readthedocs.io/en/stable/quick.html
  why: Quickstart guide for h5py. Shows how to read/write datasets from HDF5 files. Using `with h5py.File(...) as f:` is the standard pattern.

- url: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
  why: Documentation for the StandardScaler. This is the correct tool for zero-centered data. Pay attention to `.fit()`, `.transform()`, and `.inverse_transform()`.
```

### Current Codebase tree
```bash
sc_surrogate/
│
├── data/
│   └── raw/
│       └── space_charge_data.h5
│
├── configs/
│   ├── generation_config.yaml
│   └── distgen_template.yaml
│
├── generation/
│   └── generate_data.py
│
├── preprocessing/
├── modeling/
├── evaluation/
├── README.md
└── .gitignore
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
sc_surrogate/
│
├── configs/
│   ├── generation_config.yaml
│   ├── distgen_template.yaml
│   └── training_config.yaml      # ADDED: Controls all ML pipeline parameters
│
├── data/
│   ├── raw/
│   └── processed/                # CREATED: Dir for processed train/val/test data
│
├── preprocessing/
│   └── preprocess_data.py        # ADDED: Cleans, normalizes, and splits raw data
│
├── modeling/
│   └── dataset.py                # ADDED: PyTorch Dataset for loading processed data
│
└── saved_models/                 # CREATED: Dir for model checkpoints and scalers
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Do NOT use MinMaxScaler for the electric field data. It ranges from large negative to large positive values and is centered around zero. StandardScaler is the correct choice, as it preserves the zero-mean property and scales to unit variance.

# CRITICAL: HDF5 datasets should be lazy-loaded in the PyTorch Dataset. Do not load the entire dataset into memory in the `__init__` method. Open the file in `__init__` and read specific indices in `__getitem__`.

# CRITICAL: The raw HDF5 file from `generate_data.py` uses a group-per-sample structure (e.g., `run_00001/rho`). This is inefficient for training. The `preprocess_data.py` script's primary job is to read from this format and write the processed data into a monolithic format (e.g., a single `charge_density` dataset of shape `(N, Nx, Ny, Nz)`).

# CRITICAL: Data leakage is a serious risk. Scalers MUST be fit ONLY on the training data.

# CRITICAL: PyTorch models typically expect tensors of type `torch.float32`. Data read from h5py/numpy is often `float64`, so explicit conversion (`.astype('float32')` or `torch.tensor(..., dtype=torch.float32)`) is required.

# HINT: Use `joblib` or `pickle` to save the scikit-learn scaler objects for later use during inference. `joblib` is generally preferred for objects containing large numpy arrays.

# HINT: Reading all data into memory at once to fit the scaler might be memory-intensive. For now, this is acceptable, but for extremely large datasets, a strategy using `StandardScaler.partial_fit` in batches could be considered in the future.
```

## Implementation Blueprint

### Data models and structure

**Raw Data (`data/raw/simulations.h5`):**
-   File Root
    -   `run_00000` (h5py.Group)
        -   `rho` (Dataset, shape `(Nx, Ny, Nz)`)
        -   `efield` (Dataset, shape `(3, Nx, Ny, Nz)`)
        -   `parameters` (Dataset, input sigmas)
    -   `run_00001` (h5py.Group)
        -   ...
    -   ...

**Processed Data (`data/processed/train.h5`, `val.h5`, `test.h5`):**
-   File Root
    -   `charge_density` (Dataset, shape `(N_split, Nx, Ny, Nz)`)
    -   `electric_field` (Dataset, shape `(N_split, 3, Nx, Ny, Nz)`)

### list of tasks to be completed
```yaml
- Task 1:
  CREATE configs/training_config.yaml:
    - Include sections for `paths`, `preprocessing`, `model`, and `training`.
    - Ensure paths point to the correct locations in the desired file tree.

- Task 2:
  CREATE preprocessing/preprocess_data.py:
    - Load the YAML config.
    - Load the raw HDF5 file.
    - Get the total number of samples and create a shuffled array of indices.
    - Split the indices into train, validation, and test sets.
    - **CRITICAL**: Isolate the training data using the training indices.
    - **Fit Scalers**:
        - Initialize empty lists: `train_inputs`, `train_targets`.
        - Iterate through the **training keys only**. For each key:
            - Read the `rho` and `efield` data.
            - Append them to `train_inputs` and `train_targets`.
        - Concatenate the lists into large numpy arrays.
        - Reshape these arrays to be 2D for the scaler: `(n_samples * nx * ny * nz, 1)` for input and `(n_samples * 3 * nx * ny * nz, 1)` for target.
        - Fit two `StandardScaler` objects, one for inputs and one for targets.
        - Save the scalers to disk using `joblib`.
    - **Process and Save Splits**:
        - Loop through the three key lists (train, val, test) and their names ('train', 'val', 'test').
        - For each split:
            - Create a new processed HDF5 file (e.g., `data/processed/train.h5`).
            - Create empty, resizable HDF5 datasets `charge_density` and `electric_field` in the new file with the correct shape, but starting at size 0 on the first axis.
            - Iterate through the keys for that split:
                - Read the raw `rho` and `efield` data.
                - Reshape and transform the data using the **already-fitted** scalers.
                - Reshape the normalized data back to its grid format.
                - Resize the HDF5 datasets and append the new normalized sample.

- Task 3:
  CREATE modeling/dataset.py:
    - Import `torch` and `h5py`.
    - Define a class `SpaceChargeDataset(torch.utils.data.Dataset)`.
    - In `__init__(self, h5_path)`:
      - Store `h5_path`.
      - Open the HDF5 file and store file object `self.h5_file`.
      - Get handles to the input and target datasets (e.g., `self.inputs = self.h5_file['charge_density']`).
      - Store the number of samples in `self.num_samples`.
    - In `__len__(self)`:
      - Return `self.num_samples`.
    - In `__getitem__(self, idx)`:
      - Get the input and target data for the given `idx` from the H5 datasets.
      - Convert the numpy arrays to `torch.float32` tensors.
      - The charge density input needs a "channel" dimension for PyTorch CNNs. Unsqueeze it to shape `(1, Nx, Ny, Nz)`.
      - The field target should already be in the correct shape `(3, Nx, Ny, Nz)`.
      - Return `(input_tensor, target_tensor)`.
```

## Validation Loop
### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
uv run ruff check --fix

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
Create a new test file `tests/test_data_pipeline.py`. Before running, create a small, fake `data/raw/space_charge_data.h5` for testing purposes.

```python
# CREATE tests/test_data_pipeline.py
import torch
import h5py
import numpy as np
import os
import yaml

@pytest.fixture(scope="module")
def setup_pipeline_test_data():
    """[UPDATE] Creates a fake raw data file with the group-per-sample structure."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # Create fake raw data
    raw_path = "data/raw/test_simulations.h5"
    with h5py.File(raw_path, 'w') as f:
        for i in range(100): # 100 samples
            group = f.create_group(f'run_{i:05d}')
            group.create_dataset('rho', data=np.random.rand(32, 32, 32))
            group.create_dataset('efield', data=(np.random.randn(3, 32, 32, 32)))

    # Create fake config
    config_path = "configs/test_training_config.yaml"
    config = {
        'paths': {'raw_data_path': raw_path, 'processed_dir': 'data/processed/', 'model_save_dir': 'saved_models/'},
        'preprocessing': {'split_ratios': [0.8, 0.1, 0.1], 'normalization_method': 'standard'},
    }
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    yield config_path
    # ... teardown logic ...

def test_preprocess_data_script(setup_pipeline_test_data):
    """Tests that the preprocessor correctly reads the grouped format and writes a flat format."""
    config_path = setup_pipeline_test_data
    from preprocessing.preprocess_data import main as preprocess_main
    preprocess_main(config_path)

    # Assert files were created
    assert os.path.exists("data/processed/train.h5")
    assert os.path.exists("saved_models/scalers.pkl")

    # Assert the OUTPUT format is monolithic
    with h5py.File("data/processed/train.h5", 'r') as f:
        assert 'charge_density' in f
        assert 'electric_field' in f
        assert f['charge_density'].shape == (80, 32, 32, 32) # 80% of 100 samples
        assert f['electric_field'].shape == (80, 3, 32, 32, 32)
        
        # Assert normalization
        field_data = f['electric_field'][:]
        assert np.isclose(field_data.mean(), 0.0, atol=1e-5)
        assert np.isclose(field_data.std(), 1.0, atol=1e-5)
        
def test_space_charge_dataset(setup_pipeline_test_data):
    """Tests the Dataset class on the PROCESSED data file."""
    # (This test remains the same, as it validates the final output format)
    from modeling.dataset import SpaceChargeDataset
    train_dataset = SpaceChargeDataset(h5_path="data/processed/train.h5")
    assert len(train_dataset) == 80
    # ... rest of assertions
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_data_pipeline.py -v
# If failing: Read error, understand root cause, fix code, re-run.
```

### Final validation Checklist
- [ ] All tests in `tests/test_data_pipeline.py` pass.
- [ ] No linting errors: `uv run ruff check preprocessing/ modeling/`
- [ ] No type errors: `uv run mypy preprocessing/ modeling/`
- [ ] `preprocess_data.py` runs without error on the test data.
- [ ] The generated `scalers.pkl` file can be loaded with `joblib`.
- [ ] The `SpaceChargeDataset` works correctly with a `DataLoader`.

---

## Anti-Patterns to Avoid
-   ❌ Don't try to make the `Dataset` class read the raw group-per-sample file. It's inefficient. Perform the format conversion in `preprocess_data.py`.
-   ❌ Don't fit the scaler on the whole dataset. **This is the #1 mistake.** Fit on training data ONLY.
-   ❌ Don't load the entire HDF5 file into RAM in the `Dataset`'s `__init__`. This will fail with large datasets.
-   ❌ Don't hardcode paths or parameters. Read everything from the `training_config.yaml` file.
-   ❌ Don't forget to add the channel dimension to the input tensor for CNN compatibility.
-   ❌ Don't mix up the shapes of the input (charge density) and target (electric field) data.