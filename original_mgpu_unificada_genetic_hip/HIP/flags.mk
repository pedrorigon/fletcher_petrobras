CC=$(GCC)
CFLAGS=-O3
HIPCC=$(HIPCC) # Use HIPCC for AMD ROCm
HIPCFLAGS=-O3 --amdgpu-target=$(AMD_GPU_TARGET) # Adjust AMD GPU target as needed
LIBS=-L$(ROCM_PATH)/lib -lamdhip64 -lstdc++ $(GCC_LIBS) # Adjust ROCm path and libraries
