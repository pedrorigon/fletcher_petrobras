# Wave Propagation Simulation using C and CUDA/OpenMP/OpenACC

## Project Overview
This project implements a 3D wave propagation simulation using C and parallel computing techniques such as CUDA, OpenMP, or OpenACC. The simulation propagates waves through a cubic grid domain with 8th-order derivatives for high accuracy.

## Features
- **Parallelized Wave Propagation:**
  - CUDA for GPU acceleration.
  - OpenMP/OpenACC for multi-threaded CPU execution.
- **Memory Optimization:** Efficient allocation and deallocation of GPU/CPU memory.
- **Simulation Output:**
  - Optionally generates `.rsf` output files for further analysis.
  - Produces a 2D visualization video of the wave propagation.
- **Configurable Parameters:** Grid size, simulation time, and output options.

## Prerequisites
Before running the simulation, ensure you have the following dependencies installed:

### For CUDA version:
- NVIDIA CUDA Toolkit
- Compatible GPU with CUDA support
- GCC/G++ compiler

### For OpenMP/OpenACC version:
- GCC/G++ compiler with OpenMP support
- PGI or GCC compiler with OpenACC support

## Performance Optimization
The following optimization techniques are implemented to achieve high performance:
- **CUDA:** Efficient kernel launches, memory coalescing, and shared memory usage.
- **OpenMP:** Dynamic scheduling, workload balancing.
- **OpenACC:** Offloading to accelerators with memory management optimization.

## Known Issues
- Ensure proper GPU/CPU resource availability before running large simulations.
- For OpenACC, compatibility may vary across compilers.
