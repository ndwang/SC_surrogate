import h5py
from juliacall import Main as jl
import yaml
import os
import threading
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from distgen import Generator

class SpaceChargeDataGenerator:
    """
    Class for generating space charge data using distgen and Julia.
    Call generate_sigma_samples() before run() to set up the sigma samples.
    Optionally, use from_config() to load settings from a YAML file.
    """
    def __init__(self,
                 template_file,
                 n_samples=1000,
                 grid_size=(32, 32, 32),
                 min_bound=(-0.025, -0.025, -0.025),
                 max_bound=(0.025, 0.025, 0.025),
                 output_dir="data/raw",
                 output_filename="simulations.h5",
                 sigma_mins=(0.002, 0.002, 0.002),
                 sigma_maxs=(0.007, 0.007, 0.007),
                 device='cpu',
                 seed=None):
        
        # --- I/O Configuration ---
        self.template_file = template_file
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, output_filename)

        # --- Configuration ---
        self.n_samples = n_samples
        self.grid_size = tuple(grid_size)
        self.device = device

        # --- Particle Generator ---
        self.generator = self._create_generator()
        self.n_particles = self.generator["n_particle"]
        
        # --- Sampling ---
        self.sampler = qmc.LatinHypercube(d=len(sigma_mins), seed=seed)
        self.sigma_samples = qmc.scale(self.sampler.random(n=self.n_samples), sigma_mins, sigma_maxs)
        
        # --- Julia Setup ---
        jl.seval("using SpaceCharge")
        if self.device == 'gpu':
            print("Setting up for GPU (CUDA)...")
            jl.seval("using CUDA")
            self.array_type = jl.CuArray
        else:
            print("Setting up for CPU...")
            self.array_type = jl.Array
        
        self.mesh = jl.Mesh3D(tuple(grid_size), tuple(min_bound), tuple(max_bound), array_type=self.array_type)
        self._deposit_func = jl.seval("deposit!")
        self._solve_func = jl.seval("solve!")
        
        # --- Resource Management ---
        self.io_executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.h5_file = None # Will be opened in __enter__
                
        # --- Pre-allocated Julia Arrays ---
        self._particles_x_jl = self.array_type(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_y_jl = self.array_type(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_z_jl = self.array_type(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_q_jl = self.array_type(np.zeros(self.n_particles, dtype=np.float64))

    @classmethod
    def from_config(cls, config_path):
        """Create an instance from a YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def __enter__(self):
        """Set up resources: create output directory and open HDF5 file."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.h5_file = h5py.File(self.output_path, 'w')
        # Store metadata in the HDF5 file itself
        self.h5_file.attrs['n_samples'] = self.n_samples
        self.h5_file.attrs['n_particles'] = self.n_particles
        self.h5_file.attrs['grid_size'] = self.grid_size
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources: shut down thread pool and close HDF5 file."""
        print("\nWaiting for all write operations to complete...")
        self.io_executor.shutdown(wait=True)
        if self.h5_file:
            self.h5_file.close()
        print(f"Data saved to {self.output_path}")

    def _create_generator(self):
        """Create the initial generator with default values"""
        return Generator(self.template_file)

    def _update_jl_arrays(self, particles_x, particles_y, particles_z, particles_q):
        """Update pre-allocated Julia arrays with new particle data."""
        jl.seval("copyto!")(self._particles_x_jl, jl.Array(particles_x))
        jl.seval("copyto!")(self._particles_y_jl, jl.Array(particles_y))
        jl.seval("copyto!")(self._particles_z_jl, jl.Array(particles_z))
        jl.seval("copyto!")(self._particles_q_jl, jl.Array(particles_q))

    def _save_sample_to_h5(self, i, sigmas, rho, efield):
        """Saves a single sample (inputs and outputs) to the HDF5 file asynchronously."""
        def save_func():
            with self.lock:
                run_group = self.h5_file.create_group(f'run_{i:05d}')
                run_group.create_dataset('parameters', data=sigmas)
                run_group.create_dataset('rho', data=rho, compression='gzip')
                run_group.create_dataset('efield', data=efield, compression='gzip')
        self.io_executor.submit(save_func)

    def generate_sample(self, i):
        sigmas = self.sigma_samples[i]

        self.generator["x_dist:sigma_x:value"] = sigmas[0]
        self.generator["y_dist:sigma_y:value"] = sigmas[1]
        self.generator["z_dist:sigma_z:value"] = sigmas[2]

        pg = self.generator.run()

        self._update_jl_arrays(
            pg['x'].astype(np.float64),
            pg['y'].astype(np.float64),
            pg['z'].astype(np.float64),
            pg['weight'].astype(np.float64)
        )

        self._deposit_func(self.mesh, self._particles_x_jl, self._particles_y_jl, 
                          self._particles_z_jl, self._particles_q_jl)
        self._solve_func(self.mesh)

        rho = np.array(self.mesh.rho)
        efield = np.array(self.mesh.efield)

        self._save_sample_to_h5(i, sigmas, rho, efield)

    def run(self):
        for i in tqdm(range(self.n_samples)):
            self.generate_sample(i)


# If run as a script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate space charge data.")
    parser.add_argument('config', type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with SpaceChargeDataGenerator.from_config(args.config) as generator:
        generator.run()