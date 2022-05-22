#include "cudaUtilities.h"
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void floatToDoubleVector(float *left, double *right)
{
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    right[position] = left[position];
}
