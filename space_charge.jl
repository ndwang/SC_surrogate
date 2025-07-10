using SpaceCharge
using CUDA

# Define grid size
grid_size = (32, 32, 32)
min_bound = -0.025
max_bound = 0.025

# Define particle positions and charges on the GPU
particles_x_gpu = CuArray([0.0])
particles_y_gpu = CuArray([0.0])
particles_z_gpu = CuArray([0.0])
particles_q_gpu = CuArray([1.0e-9])

# Create a Mesh3D object with automatic bounds on the GPU (array_type=CuArray)
mesh_gpu = Mesh3D(grid_size, min_bound, max_bound; array_type=CuArray)

# Deposit particle charges onto the grid (on GPU)
deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)

# Solve for electric field (on GPU)
solve!(mesh_gpu)

