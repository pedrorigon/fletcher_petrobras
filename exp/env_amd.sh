#!/usr/bin/env bash

# Choose GCC or other appropriate compiler
export GCC=hipcc
export OMP_FLAG=-fopenmp

# Define the number of OpenMP threads based on your system's CPU cores
export OMP_NUM_THREADS=$(nproc)

# Set HIP-related environment variables
export HIP_PLATFORM=amd
export HIP_DEVICE=gfx908  # Adjust this to the appropriate AMD GPU architecture

# If you need to specify specific ROCm paths or libraries, add them here
# export HIP_DIR=/path/to/rocm/hip
# export LD_LIBRARY_PATH=$HIP_DIR/lib:$LD_LIBRARY_PATH

# If you want to target a specific AMD GPU architecture, adjust HIP_DEVICE accordingly.
# Use 'hipconfig' to list available target architectures:
# hipconfig --print-hip-arch

# For example, to target gfx906 (MI100):
# export HIP_DEVICE=gfx906

# To target a specific number of threads for HIP kernels (optional):
# export HIP_NUM_THREADS=64

# To enable ROCm's HSA (Heterogeneous System Architecture) runtime (optional):
# export HSA_ENABLE_SDMA=0
# export HSA_AMDGPU_OPTS=--unified
