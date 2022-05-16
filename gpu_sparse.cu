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
#include "cudaUtilities.h"

extern "C"
{

    #include "utilities.h"
    #include "read.h"
}


void solveSystemSparse(SparseMatrix* mat, Vector *B, double* X)
{
    double *Xcalculated = (double *)malloc(mat->size * sizeof(double));

    // INITIALIZE CUSOLVER
    cusolverSpHandle_t cusolverHandle;
    cudaStream_t stream = NULL;
    cusolverStatus_t error;

    cusolverSpCreate(&cusolverHandle);
    cudaStreamCreate(&stream);
    cusolverSpSetStream(cusolverHandle, stream);

    int n = mat->size;
    int nnz = mat->row_idx[n];
    double tolerance = 1e-12;
    int reorder = 0;
    int *singularity = (int *)malloc(sizeof(int));

    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // TRANSFER DATA TO GPU
    // double *csrValA, *b, *x;
    // int *csrRowPtrA, *csrColIndA;

    // cudaMalloc((void **)&csrValA, nnz * sizeof(double));
    // cudaMalloc((void **)&b, n * sizeof(double));
    // cudaMalloc((void **)&csrRowPtrA, (n + 1) * sizeof(int));
    // cudaMalloc((void **)&csrColIndA, nnz * sizeof(int));

    // cudaMemcpy(csrValA, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(b, B->values, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(csrRowPtrA, mat->row_idx, (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(csrColIndA, mat->col_idx, nnz * sizeof(double), cudaMemcpyHostToDevice);

    error = cusolverSpDcsrlsvluHost(
        cusolverHandle,
        n,
        nnz,
        descrA,
        mat->values,
        mat->row_idx,
        mat->col_idx,
        B->values,
        tolerance,
        reorder,
        X,
        singularity);


    // cudaFree(csrValA);
    // cudaFree(b);
    // cudaFree(x);
    // cudaFree(csrRowPtrA);
    // cudaFree(csrColIndA);

    cusolverSpDestroy(cusolverHandle);

    //printf("Singularity is %d\n",singularity);
    //printf("Status is %s\n",cudaGetErrorEnum(error));
}

int main(int argc, char** argv){
     char *matrixName;
    if (argc == 1)
        matrixName = "data/e20r0000";
    else
        matrixName = argv[1];
    char *filename = (char *)malloc(40 * sizeof(char));
    char *filenameB = (char *)malloc(40 * sizeof(char));
    char *filenameSol = (char *)malloc(40 * sizeof(char));

    sprintf(filename, "%s.mtx", matrixName);
    sprintf(filenameB, "%s_rhs1.mtx", matrixName);

    SparseMatrix *sparse = (SparseMatrix *)malloc(sizeof(SparseMatrix));
    Vector *B = (Vector *)malloc(sizeof(Vector));

    readSparseMMMatrix(filename, sparse);
    readMMVector(filenameB, B);

    double *X = (double *)malloc(B->size * sizeof(double));

    struct timeval start = tic();

    solveSystemSparse(sparse, B, X);

    printf("Sparse time is %f\n", toc(start));

    readSparseMMMatrix(filename, sparse);    // overwrite factorized matrix to get original values for evaluation
    checkSolutionSparse(sparse, B, X, 0); // calculate |Ax-b|
    saveVector("Sparse.txt",B->size,X);

}