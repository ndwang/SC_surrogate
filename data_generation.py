from juliacall import Main as jl
import numpy as np
from scipy.stats import qmc
import os
from tqdm import tqdm
from distgen import Generator
import yaml

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
                 sigma_maxs=[0.007, 0.007, 0.007]):
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
        os.makedirs(self.output_dir, exist_ok=True)
        # Julia setup
        jl.seval("using SpaceCharge, CUDA")
        self.mesh = jl.Mesh3D(self.grid_size, self.min_bound, self.max_bound, array_type=jl.CuArray)
        # Sampler
        self.sampler = qmc.LatinHypercube(d=3)
        self.sigma_samples = None

    @classmethod
    def from_config(cls, config_path):
        """
        Create an instance from a YAML config file.
        Example YAML keys: n_samples, n_particles, total_charge, grid_size, min_bound, max_bound, output_dir, sigma_mins, sigma_maxs
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def generate_sigma_samples(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        unit_samples = self.sampler.random(n=self.n_samples)
        self.sigma_samples = qmc.scale(unit_samples, self.sigma_mins, self.sigma_maxs)

    def generate_template(self, sigma_x, sigma_y, sigma_z):
        return f"""
        n_particle: {self.n_particles}
        random:
            type: hammersley
        start:
            tstart: 0 sec
            type: time
        x_dist:
            sigma_x: {sigma_x} mm
            n_sigma_cutoff: 3
            type: gaussian
        y_dist:
            sigma_y: {sigma_y} mm
            n_sigma_cutoff: 3
            type: gaussian
        z_dist:
            sigma_z: {sigma_z} mm
            n_sigma_cutoff: 3
            type: gaussian
        total_charge: {self.total_charge} C
        species: electron
        """

    def generate_sample(self, i):
        if self.sigma_samples is None:
            raise ValueError("sigma_samples not generated. Call generate_sigma_samples() first.")
        sigma_x, sigma_y, sigma_z = self.sigma_samples[i]
        template = self.generate_template(sigma_x, sigma_y, sigma_z)
        gen = Generator(template)
        pg = gen.run()
        # Move to Julia
        particles_x = pg['x'].astype(np.float64)
        particles_y = pg['y'].astype(np.float64)
        particles_z = pg['z'].astype(np.float64)
        particles_q = pg['weight'].astype(np.float64)
        particles_x_gpu = jl.CuArray(particles_x)
        particles_y_gpu = jl.CuArray(particles_y)
        particles_z_gpu = jl.CuArray(particles_z)
        particles_q_gpu = jl.CuArray(particles_q)
        # Deposit and solve
        jl.seval("deposit!")(self.mesh, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
        jl.seval("solve!")(self.mesh)
        efield = np.array(self.mesh.efield)
        # Save data
        np.save(f"{self.output_dir}/efield_{i:04d}.npy", efield)

    def run(self):
        if self.sigma_samples is None:
            self.generate_sigma_samples()
        for i in tqdm(range(self.n_samples)):
            self.generate_sample(i)

# If run as a script
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generator = SpaceChargeDataGenerator.from_config(sys.argv[1])
    else:
        generator = SpaceChargeDataGenerator()
    generator.generate_sigma_samples()
    generator.run()