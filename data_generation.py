from juliacall import Main as jl
import numpy as np
from scipy.stats import qmc
import os
from tqdm import tqdm
from distgen import Generator
import yaml
from concurrent.futures import ThreadPoolExecutor
import threading

class SpaceChargeDataGenerator:
    """
    Class for generating space charge data using Julia and distgen.
    Call generate_sigma_samples() before run() to set up the sigma samples.
    Optionally, use from_config() to load settings from a YAML file.
    """
    def __init__(self,
                 n_samples=1000,
                 n_particles=100_000,
                 total_charge=1.0,
                 grid_size=(32, 32, 32),
                 min_bound=(-0.025, -0.025, -0.025),
                 max_bound=(0.025, 0.025, 0.025),
                 output_dir="data",
                 sigma_mins=[0.002, 0.002, 0.002],
                 sigma_maxs=[0.007, 0.007, 0.007],
                 template_file=None):
        self.n_samples = n_samples
        self.n_particles = n_particles
        self.total_charge = total_charge
        self.charge_per_particle = total_charge / n_particles
        self.grid_size = tuple(grid_size)
        self.min_bound = tuple(min_bound)
        self.max_bound = tuple(max_bound)
        self.output_dir = output_dir
        self.sigma_mins = list(sigma_mins)
        self.sigma_maxs = list(sigma_maxs)
        self.template_file = template_file
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Julia setup - cache functions to avoid repeated seval calls
        jl.seval("using SpaceCharge, CUDA")
        self.mesh = jl.Mesh3D(self.grid_size, self.min_bound, self.max_bound, array_type=jl.CuArray)
        
        # Cache Julia functions
        self._deposit_func = jl.seval("deposit!")
        self._solve_func = jl.seval("solve!")
        
        # Pre-allocate GPU arrays for particle data
        self._particles_x_gpu = jl.CuArray(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_y_gpu = jl.CuArray(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_z_gpu = jl.CuArray(np.zeros(self.n_particles, dtype=np.float64))
        self._particles_q_gpu = jl.CuArray(np.zeros(self.n_particles, dtype=np.float64))
        
        # Sampler
        self.sampler = qmc.LatinHypercube(d=3)
        self.sigma_samples = None
        
        # Thread pool for I/O operations
        self.io_executor = ThreadPoolExecutor(max_workers=4)
        
        # Create the generator based on the template
        self.generator = self._create_generator()

    @classmethod
    def from_config(cls, config_path):
        """
        Create an instance from a YAML config file.
        Example YAML keys: n_samples, n_particles, total_charge, grid_size, min_bound, max_bound, output_dir, sigma_mins, sigma_maxs, template_file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def generate_sigma_samples(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        unit_samples = self.sampler.random(n=self.n_samples)
        self.sigma_samples = qmc.scale(unit_samples, self.sigma_mins, self.sigma_maxs)

    def _get_template(self):
        """Get the distgen template with current parameters"""
        if self.template_file and os.path.exists(self.template_file):
            # Load template from file
            with open(self.template_file, 'r') as f:
                template = f.read()
            return template
        else:
            # Fallback to inline template
            return f"""
            n_particle: {self.n_particles}
            random:
                type: hammersley
            start:
                tstart: 0 sec
                type: time
            x_dist:
                sigma_x: 0.005 m
                n_sigma_cutoff: 3
                type: gaussian
            y_dist:
                sigma_y: 0.005 m
                n_sigma_cutoff: 3
                type: gaussian
            z_dist:
                sigma_z: 0.005 m
                n_sigma_cutoff: 3
                type: gaussian
            total_charge: {self.total_charge} C
            species: electron
            """

    def _create_generator(self):
        """Create the initial generator with default values"""
        template = self._get_template()
        return Generator(template)

    def _update_gpu_arrays(self, particles_x, particles_y, particles_z, particles_q):
        """Update pre-allocated GPU arrays with new particle data"""
        # Use Julia's copyto! to avoid reallocation
        jl.seval("copyto!")(self._particles_x_gpu, particles_x)
        jl.seval("copyto!")(self._particles_y_gpu, particles_y)
        jl.seval("copyto!")(self._particles_z_gpu, particles_z)
        jl.seval("copyto!")(self._particles_q_gpu, particles_q)

    def _save_efield_async(self, efield, filename):
        """Save efield data asynchronously"""
        def save_func():
            np.save(filename, efield)
        
        self.io_executor.submit(save_func)

    def generate_sample(self, i):
        if self.sigma_samples is None:
            raise ValueError("sigma_samples not generated. Call generate_sigma_samples() first.")
        sigma_x, sigma_y, sigma_z = self.sigma_samples[i]
        
        # Update with current sigma values
        self.generator["x_dist:sigma_x:value"] = sigma_x
        self.generator["y_dist:sigma_y:value"] = sigma_y
        self.generator["z_dist:sigma_z:value"] = sigma_z
        
        pg = self.generator.run()
        
        # Update pre-allocated GPU arrays instead of creating new ones
        self._update_gpu_arrays(
            pg['x'].astype(np.float64),
            pg['y'].astype(np.float64),
            pg['z'].astype(np.float64),
            pg['weight'].astype(np.float64)
        )
        
        # Use cached Julia functions
        self._deposit_func(self.mesh, self._particles_x_gpu, self._particles_y_gpu, 
                          self._particles_z_gpu, self._particles_q_gpu)
        self._solve_func(self.mesh)
        
        # Get efield data
        efield = np.array(self.mesh.efield)
        
        # Save data asynchronously
        filename = f"{self.output_dir}/efield_{i:04d}.npy"
        self._save_efield_async(efield, filename)

    def run(self):
        if self.sigma_samples is None:
            self.generate_sigma_samples()
        
        for i in tqdm(range(self.n_samples)):
            self.generate_sample(i)
        
        # Wait for all I/O operations to complete
        self.io_executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'io_executor'):
            self.io_executor.shutdown(wait=True)

# If run as a script
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generator = SpaceChargeDataGenerator.from_config(sys.argv[1])
    else:
        generator = SpaceChargeDataGenerator()
    generator.generate_sigma_samples()
    generator.run()