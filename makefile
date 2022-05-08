CC = gcc
NVCC = nvcc
CFLAGS = -lopenblas -O3
CUDA = -lcublas -lcusolver

all: cpu gpu cpu_sparse

cpu: cpu.c utilities.c mmio.c read.c
	$(CC) -o $@ $^ $(CFLAGS)

gpu: gpu.cu utilities.c mmio.c read.c
	$(NVCC) $^ -o $@  $(CUDA)

cpu_sparse: cpu_sparse.c utilities.c mmio.c read.c
	$(CC) -o $@ $^ $(CFLAGS)
	
