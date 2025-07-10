import numpy as np
from scipy.stats import qmc
import os
from tqdm import tqdm
from distgen import Generator
from juliacall import Main as jl

# 1. Constants setup
N_SAMPLES = 1
N_PARTICLES = 100_000
TOTAL_CHARGE = 1.0
CHARGE_PER_PARTICLE = TOTAL_CHARGE / N_PARTICLES

GRID_SIZE = (32, 32, 32)
MIN_BOUND = (-0.025, -0.025, -0.025) # -25 mm
MAX_BOUND = (0.025, 0.025, 0.025)    # +25 mm

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 2. latin Hypercube Sampling for Sigmas
sigma_mins = [0.002, 0.002, 0.002]  # 2 mm
sigma_maxs = [0.007, 0.007, 0.007]  # 7 mm
sampler = qmc.LatinHypercube(d=3)
unit_samples = sampler.random(n=N_SAMPLES)
sigma_samples = qmc.scale(unit_samples, sigma_mins, sigma_maxs)

# 3. Generation preparation

def generate_template(sigma_x, sigma_y, sigma_z):
    return f"""
    n_particle: 100000
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
    total_charge: 1 C
    species: electron
    """
# Julia setup
jl.seval("using SpaceCharge, CUDA")
mesh = jl.Mesh3D(GRID_SIZE, MIN_BOUND, MAX_BOUND, array_type=jl.CuArray)

# 4. Data Generation
for i in tqdm(range(N_SAMPLES)):
    # Beam generation
    sigma_x, sigma_y, sigma_z = sigma_samples[i]
    template = generate_template(sigma_x, sigma_y, sigma_z)
    gen = Generator(template)
    pg = gen.run()
    # Move to Julia
    particles_x = pg['x'].astype(np.float64)
    particles_y = pg['y'].astype(np.float64)
    particles_z = pg['z'].astype(np.float64)
    particles_q = pg['q'].astype(np.float64)
    particles_x_gpu = jl.CuArray(particles_x)
    particles_y_gpu = jl.CuArray(particles_y)
    particles_z_gpu = jl.CuArray(particles_z)
    particles_q_gpu = jl.CuArray(particles_q)
    # Deposit and solve
    jl.seval("deposit!")(mesh, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
    jl.seval("solve!")(mesh)
    efield = jl.np.array(mesh.efield)
    # Save data
    np.save(f"{OUTPUT_DIR}/efield_{i}.npy", efield)