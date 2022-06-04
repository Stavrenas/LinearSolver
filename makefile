CC = gcc
NVCC = nvcc
ICX = icx
CFLAGS = -g -Wall -llapacke -llapack -lm
CUDA = -lcublas -lcusolver -lcusparse -g -G
LINKER =  -L${MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl
ICXFLAGS =  -DMKL_ILP64  -qmkl=parallel
MKL = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core

all: cpu gpu gpu_sparse

cpu: cpu.c utilities.c mmio.c read.c
	$(CC) -o $@ $^ $(CFLAGS)

gpu: gpu.cu utilities.c mmio.c read.c cudaUtilities.cu
	$(NVCC) $^ -o $@  $(CUDA)

gpu_sparse: gpu_sparse.cu cudaUtilities.cu utilities.c mmio.c read.c 
	$(NVCC) -o $@ $^ $(CUDA) test.o
	
test: test.c 
	$(ICX) -c $@ $^ $(LINKER) $(ICXFLAGS) -fPIE

clean:
	rm -f cpu gpu gpu_sparse test