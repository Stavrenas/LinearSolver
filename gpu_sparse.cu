#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverSp.h>
#include "device_launch_parameters.h"
#include <cuda.h>

extern "C"
{
#include "utilities.h"
#include "read.h"
}

static const char *cudaGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";
}


int main()
{

    char *matrixName = "Test";
    char *filename = (char *)malloc(40 * sizeof(char));
    char *filenameB = (char *)malloc(40 * sizeof(char));
    char *filenameX = (char *)malloc(40 * sizeof(char));
    char *filenameSol = (char *)malloc(40 * sizeof(char));

    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

    sprintf(filename, "%s.mtx", matrixName);
    sprintf(filenameB, "%s-B.txt", matrixName);
    sprintf(filenameX, "%s-X.txt", matrixName);
    sprintf(filenameSol, "%s-Solution.txt", matrixName);

    //generateMMMatrix(filename, 100, 1000);
    readMMMatrix(filename, mat);
    // printMatrix(mat);

    double *B = (double *)malloc(mat->size * sizeof(double));
    double *X = (double *)malloc(mat->size * sizeof(double));
    double *Xcalculated = (double *)malloc(mat->size * sizeof(double));

    generateSolutionVector(matrixName, mat);
    readVector(filenameB, mat->size, B);
    readVector(filenameX, mat->size, X);

    // INITIALIZE CUSOLVER
    cusolverSpHandle_t cusolverHandle;
    cudaStream_t stream = NULL;
    cusolverStatus_t error;

    cusolverSpCreate(&cusolverHandle);
    cudaStreamCreate(&stream);
    cusolverSpSetStream(cusolverHandle, stream);

    int n = mat->size;
    int nnz = mat->row_idx[n];
    double tolerance = 1e-38;
    int reorder = 0;
    int *singularity = (int *)malloc(sizeof(int));

    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // TRANSFER DATA TO GPU
    double *csrValA, *b, *x;
    int *csrRowPtrA, *csrColIndA;

    cudaMalloc((void **)&csrValA, nnz * sizeof(double));
    cudaMalloc((void **)&b, n * sizeof(double));
    cudaMalloc((void **)&x, n * sizeof(double));
    cudaMalloc((void **)&csrRowPtrA, (n + 1) * sizeof(int));
    cudaMalloc((void **)&csrColIndA, nnz * sizeof(int));

    cudaMemcpy(csrValA, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(csrRowPtrA, mat->row_idx, (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIndA, mat->col_idx, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // cusolverSpDcsrlsvluHost(
    //     cusolverHandle,
    //     n,
    //     nnz,
    //     descrA,
    //     csrValA,
    //     csrRowPtrA,
    //     csrColIndA,
    //     b,
    //     tolerance,
    //     reorder,
    //     x,
    //     singularity);

    error = cusolverSpDcsrlsvluHost(
        cusolverHandle,
        n,
        nnz,
        descrA,
        mat->values,
        mat->row_idx,
        mat->col_idx,
        B,
        tolerance,
        reorder,
        Xcalculated,
        singularity);

    // cudaMemcpy(x, Xcalculated, sizeof(double) * n, cudaMemcpyDeviceToHost);

    saveVector(filenameSol, mat->size, Xcalculated);

    cudaFree(csrValA);
    cudaFree(b);
    cudaFree(x);
    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);

    cusolverSpDestroy(cusolverHandle);

    if (checkSolutionThres(n,Xcalculated,X,1e-5))
        printf("Solution is True\n");
    else
        printf("Solution is False\n");

    printf("Singularity is %d\n",singularity);
    printf("Status is %s\n",cudaGetErrorEnum(error));
}