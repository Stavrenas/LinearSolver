CC = gcc
NVCC = nvcc
CFLAGS = -g -Wall -llapacke -llapack
CUDA = -lcublas -lcusolver

all: cpu gpu gpu_sparse

cpu: cpu.c utilities.c mmio.c read.c
	$(CC) -o $@ $^ $(CFLAGS)

gpu: gpu.cu utilities.c mmio.c read.c cudaUtilities.cu
	$(NVCC) $^ -o $@  $(CUDA)

gpu_sparse: gpu_sparse.cu cudaUtilities.cu utilities.c mmio.c read.c 
	$(NVCC) -o $@ $^ $(CUDA) -lcusparse
	
