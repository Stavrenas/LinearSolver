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

// __global__ void matrixVectorMult(int n, int nnz, double *Avalues, int *rowPtr, int *colIdx, double *Vvalues)
// {
//     int total_index = threadIdx.x + blockIdx.x * blockDim.x;

//     int threads = 64;
//     int blocks = 512;

//     int lines_per_block = n / blocks + 1;

//     int startLine = blockIdx.x * lines_per_block;
//     int endLine = (blockIdx.x +1)* lines_per_block;
//     if (endLine > n)
//         endLine = n;

//     int lines_per_thread = lines_per_block / threads + 1;

//     if (total_index % 512 ==  0)
//         printf("total index %d, lines_per_block %d , lines_per_thread %d, blockDim.x %d, start line %d, end line %d\n",total_index, lines_per_block, lines_per_thread, blockDim.x,startLine, endLine);
//     double *results = (double *)malloc(lines_per_thread * sizeof(double));

//     double result = 0.0;
//     // double vector = Vvalues[index];
//     for (int j = 0; j < lines_per_thread; j++)
//     {
//         int line = startLine + threadIdx.x + threads * j;
//         if(line> endLine)
//             line = endLine;
//         int start = rowPtr[line];
//         int end = rowPtr[line + 1];
//         for (int i = start; i < end; i++)
//             result += Avalues[i] * Vvalues[colIdx[i]];

//         results[j] = result;
//     }

//     __syncthreads();

//     for (int j = 0; j < lines_per_thread; j++)
//     {
//         int line = startLine + threadIdx.x + threads * j;
//         Vvalues[line] = results[j];
//     }
// }

__global__ void matrixVectorMult(int n, int nnz, double *Avalues, int *rowPtr, int *colIdx, double *Vvalues, double * result)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n)
    {
        double dot = 0;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        for (int jj = row_start; jj < row_end; jj++)
            dot += Avalues[jj] * Vvalues[colIdx[jj]];
        result[row] = dot;
    }
}

__global__ void spmv_csr_vector_kernel(const int num_rows, const int *ptr, const int *indices, const double *data, const double *x, double *y)
{
    __shared__ double vals[64];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    int warp_id = thread_id / 32;                          // global warp index
    int lane = thread_id & (32 - 1);                       // thread index within the warp
    // one warp per row
    int row = warp_id;
    if (row < num_rows)
    {
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        // compute running sum per thread
        vals[threadIdx.x] = 0;
        for (int jj = row_start + lane; jj < row_end; jj += 32)
            vals[threadIdx.x] += data[jj] * x[indices[jj]];
        // parallel reduction in shared memory
        if (lane < 16)
            vals[threadIdx.x] += vals[threadIdx.x + 16];
        if (lane < 8)
            vals[threadIdx.x] += vals[threadIdx.x + 8];
        if (lane < 4)
            vals[threadIdx.x] += vals[threadIdx.x + 4];
        if (lane < 2)
            vals[threadIdx.x] += vals[threadIdx.x + 2];
        if (lane < 1)
            vals[threadIdx.x] += vals[threadIdx.x + 1];
        // first thread writes the result
        if (lane == 0)
            y[row] += vals[threadIdx.x];
    }
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

void residual(SparseMatrix *mat, Vector *B, double *X)
{
    int n = mat->size;
    int nnz = mat->row_idx[n];

    // INITIALIZE CUSOLVER AND CUBLAS
    cusparseHandle_t sparseHandle = NULL;
    cublasHandle_t blasHandle;
    cudaStream_t stream = NULL;
    // cusparseStatus_t status;

    cusparseCreate(&sparseHandle);
    cublasCreate(&blasHandle);
    cudaStreamCreate(&stream);

    double *Avalues, *rhs, *solution;
    int *rowPtr, *colIdx;
    checkCudaErrors(cudaMalloc((void **)&Avalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&solution, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&rhs, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));

    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Avalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhs, B->values, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(solution, X, n * sizeof(double), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t descrA;
    cusparseCreateCsr(&descrA, n, n, nnz, rowPtr, colIdx, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t descrX, descrB;
    cusparseCreateDnVec(&descrB, n, rhs, CUDA_R_64F);
    cusparseCreateDnVec(&descrX, n, solution, CUDA_R_64F);

    // INITIALIZE VARIABLES FOR LU SOLVE
    size_t spMvBufferSize;
    void *spMvBuffer;
    double minusOne = -1.0;
    double one = 1.0;
    double bNorm;
    cublasDnrm2(blasHandle, n, rhs, 1, &bNorm);
    checkCudaErrors(cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrA, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize));
    checkCudaErrors(cudaMalloc(&spMvBuffer, spMvBufferSize));
    checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrA, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));

    // RESIDUAL NORM
    double resNorm;
    cublasDnrm2(blasHandle, n, rhs, 1, &resNorm);
    printf("Residual norm is %e\n", resNorm / bNorm);

    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrB);
    cudaFree(Avalues);
    cudaFree(rhs);
    cudaFree(rowPtr);
    cudaFree(colIdx);
    cudaFree(solution);
    cudaFree(spMvBuffer);
}
