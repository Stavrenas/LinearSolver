#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverSp.h>
#include "device_launch_parameters.h"
#include "cudaUtilities.h"
#include "helper_cuda.h"

__global__ void floatToDoubleVector(float *left, double *right, int size)
{
    int parts = (float)size / blockDim.x + 1;
    int start = threadIdx.x * parts;
    int end = start + parts;
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

void gpuLU(SparseMatrix *mat)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];

    int maxThreads, blocks, threads;
    threads = 256;
    if (nnz > threads)
    {
        maxThreads = threads;
        blocks = nnz / maxThreads + 1;
    }
    else
    {
        blocks = 1;
        maxThreads = nnz;
    }
    blocks = 1;

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++)
    {
        host_float_values[i] = mat->values[i];
    }

    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle = NULL;
    // cudaStream_t stream = NULL;
    // cusparseStatus_t status;

    cusparseCreate(&sparseHandle);
    // cudaStreamCreate(&stream);

    // ALLOCATE MEMORY
    float *f_values;
    double *d_values;
    int *rowPtr, *colIdx;

    checkCudaErrors(cudaMalloc((void **)&f_values, nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_values, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));

    // COPY MATRIX A TO DEVICE MEMORY
    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // COPY FLOAT MATRIX ELEMENTS
    checkCudaErrors(cudaMemcpy(f_values, host_float_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // INITIALIZE VARIABLES FOR LU FACTORIZATION
    int pBufferSize;
    void *pBuffer;
    // int structural_zero, numerical_zero;

    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    csrilu02Info_t LUinfo;
    cusparseCreateCsrilu02Info(&LUinfo);

    double tole = 0;
    float boost = 1e-8;
    checkCudaErrors(cusparseScsrilu02_numericBoost(sparseHandle, LUinfo, 1, &tole, &boost));

    // printf("Buffer size..\n");
    // CALCULATE LU FACTORIZATION BUFFER SIZE

    checkCudaErrors(cusparseScsrilu02_bufferSize(sparseHandle, n, nnz, descrA,
                                                 f_values, rowPtr, colIdx, LUinfo, &pBufferSize));

    checkCudaErrors(cudaMalloc(&pBuffer, pBufferSize));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes
    // printf("Buffer size for LU is %d\n", pBufferSize);

    // printf("Analysis..\n");
    // LU FACTORIZATION ANALYSIS
    checkCudaErrors(cusparseScsrilu02_analysis(sparseHandle, n, nnz, descrA,
                                               f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    cusparseStatus_t status;
    int structural_zero;
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);

    // printf("Factorization..\n");
    // A = L * U
    checkCudaErrors(cusparseScsrilu02(sparseHandle, n, nnz, descrA,
                                      f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    // GPU TYPECAST
    floatToDoubleVector<<<blocks, maxThreads>>>(f_values, d_values, nnz);

    cudaMemcpy(mat->values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(f_values);
    cudaFree(d_values);
    cudaFree(rowPtr);
    cudaFree(colIdx);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroyCsrilu02Info(LUinfo);
    cusparseDestroy(sparseHandle);
}

void sortSparseMatrix(SparseMatrix *mat)
{
    int n = mat->size;
    int nnz = mat->row_idx[n];

    double *values;
    int *rowPtr, *colIdx;
    checkCudaErrors(cudaMalloc((void **)&values, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(values, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    double *temp_values;
    int *temp_rowPtr, *temp_colIdx;
    checkCudaErrors(cudaMalloc((void **)&temp_values, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_colIdx, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&temp_rowPtr, (n + 1) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(temp_values, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle = NULL;
    cusparseCreate(&sparseHandle);

    size_t bufferSize;
    void *buffer;
    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(sparseHandle, n, n, nnz, values, rowPtr, colIdx, temp_values,
                                                  temp_rowPtr, temp_colIdx, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                                  CUSPARSE_CSR2CSC_ALG1, &bufferSize));

    // printf("Buffer size is %d\n", bufferSize);
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    checkCudaErrors(cusparseCsr2cscEx2(sparseHandle, n, n, nnz, values, rowPtr, colIdx, temp_values,
                                       temp_rowPtr, temp_colIdx, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                       CUSPARSE_CSR2CSC_ALG1, buffer));

    // RUN TWICE TO GET SORTED MATRIX
    checkCudaErrors(cusparseCsr2cscEx2(sparseHandle, n, n, nnz, temp_values, temp_rowPtr, temp_colIdx, values,
                                       rowPtr, colIdx, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                       CUSPARSE_CSR2CSC_ALG1, buffer));

    checkCudaErrors(cudaMemcpy(mat->values, values, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mat->row_idx, rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mat->col_idx, colIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    cusparseDestroy(sparseHandle);
    cudaFree(values);
    cudaFree(rowPtr);
    cudaFree(colIdx);
    cudaFree(temp_values);
    cudaFree(temp_rowPtr);
    cudaFree(temp_colIdx);

    cudaFree(buffer);
}
