import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to the processed training set
h5_path = "data/processed/train.h5"

# Open the HDF5 file and extract the first sample's electric field
with h5py.File(h5_path, 'r') as f:
    # Shape: (N, 3, Nx, Ny, Nz)
    electric_field = f['electric_field'][0]  # shape: (3, Nx, Ny, Nz)

# Flatten all components into a 1D array
efield_flat = electric_field.flatten()

# Print basic statistics
print(f"Electric field stats:\n  min: {efield_flat.min()}\n  max: {efield_flat.max()}\n  mean: {efield_flat.mean()}\n  std: {efield_flat.std()}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(efield_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Electric Field Values (First Training Sample)')
plt.xlabel('Electric Field Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
