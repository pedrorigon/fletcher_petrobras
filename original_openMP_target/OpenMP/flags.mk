CC=$(GCC)
CFLAGS=-O3 -fopenmp -foffload=nvptx-none -fcf-protection=none -foffload=-misa=sm_35 -fno-stack-protector $(OMP_FLAG)
LIBS=$(GCC_LIBS)

