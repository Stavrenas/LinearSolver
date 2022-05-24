#include "cudaUtilities.h"
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void floatToDoubleVector(float *left, double *right, int size)
{
    int parts = (float)size / blockDim.x + 1;
    int start = threadIdx.x * parts;
    int end = start + parts ;
    if (end > size)
        end = size;
    for (int i = start; i < end; i++)
        right[i] = left[i];
}

// __global__ void floatToDoubleVector(float *left, double *right, int size)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index <size)
//     right[index] = left[index];
// }