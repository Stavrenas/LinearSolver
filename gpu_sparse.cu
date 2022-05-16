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
#include <cusolverRf.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cudaUtilities.h"

extern "C"
{
#include "utilities.h"
#include "read.h"
}

void solveSystemSparse(SparseMatrix *mat, Vector *B, double *X)
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
    double tol = 1e-8;
    int reorder = 3;
    int singularity;

    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    error = cusolverSpDcsrlsvluHost(
        cusolverHandle,
        n,
        nnz,
        descrA,
        mat->values,
        mat->row_idx,
        mat->col_idx,
        B->values,
        tol, // tolerance to determine matrix singularity
        reorder,
        X,
        &singularity);

    cusolverSpDestroy(cusolverHandle);

    printf("Singularity is %d\n", singularity);
    // printf("Status is %s\n",cudaGetErrorEnum(error));
}

void solveSystemSparseIterative(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{
    //create float copy of values
    float *host_float_values = (float *)malloc(mat->size * sizeof(float));
    for(int i =0 ; i< mat->size; i++)
        host_float_values[i] = mat->values[i];
    
    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle;
    cublasHandle_t blasHandle;
    cudaStream_t stream = NULL;

    cusparseStatus_t status;

    cusparseCreate(&sparseHandle);
    cublasCreate(&blasHandle);
    cudaStreamCreate(&stream);

    int n = mat->size;
    int nnz = mat->row_idx[n];
    double tol = 1e-12;
    int reorder = 0;
    int *singularity = (int *)malloc(sizeof(int));

    //ALLOCATE SPACE

    double *Xcalculated = (double *)malloc(n * sizeof(double));
    double *d_values,*rhs, *solution, *temp_solution,*deviceB;
    float *f_values;
    int *rowPtr, *colIdx;

    cudaMalloc((void **)&d_values, n * sizeof(double));
    cudaMalloc((void **)&rhs, n * sizeof(double));
    cudaMalloc((void **)&solution, n * sizeof(double));

    cudaMalloc((void **)&temp_solution, n * sizeof(double));
    cudaMalloc((void **)&f_values, n * sizeof(float));

    cudaMalloc((void **)&rowPtr, n* sizeof(int));
    cudaMalloc((void **)&colIdx, n* sizeof(int));
    cudaMalloc((void **)&deviceB, n * sizeof(double));

    cudaMemcpy(d_values, mat->values, n, cudaMemcpyHostToDevice);
    cudaMemcpy(f_values, host_float_values, n, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    cudaMemcpy(rowPtr, mat->row_idx, n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->values, n, cudaMemcpyHostToDevice);
    cudaMemcpy(colIdx, mat->col_idx, n, cudaMemcpyHostToDevice);


    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // SETUP LU FACTORIZATION
    csrilu02Info_t LUinfo = 0;
    cusparseCreateCsrilu02Info(&LUinfo);
    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    double norm;
    int max, min;

    cublasDnrm2(blasHandle, n, deviceB, 1, &norm);
    cublasIdamax(blasHandle, n, deviceB, 1, &max);
    cublasIdamin(blasHandle, n, deviceB, 1, &min);
    printf("Norm is %f, max is %f, min is %f\n", norm, B->values[max], B->values[min]);

    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrsv2Info_t info_L = 0;
    csrsv2Info_t info_U = 0;
    int pBufferSize_A;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha = 1.;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix L is base-0
    // - matrix L is lower triangular
    // - matrix L has unit diagonal
    // - matrix U is base-0
    // - matrix U is upper triangular
    // - matrix U has non-unit diagonal

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateCsrilu02Info(&LUinfo);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseDcsrilu02_bufferSize(sparseHandle, n, nnz,
                                 descrA, d_values, rowPtr, colIdx, LUinfo, &pBufferSize_A);
    cusparseDcsrsv2_bufferSize(sparseHandle, trans_L, n, nnz,
                               descr_L, d_values, rowPtr, colIdx, info_L, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(sparseHandle, trans_U, n, nnz,
                               descr_U, d_values, rowPtr, colIdx, info_U, &pBufferSize_U);

    pBufferSize = fmax(pBufferSize_A, fmax(pBufferSize_L, pBufferSize_U));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on A
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on U
    // The lower(upper) triangular part of A has the same sparsity pattern as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    cusparseDcsrilu02_analysis(sparseHandle, n, nnz, descrA,
                               d_values, rowPtr, colIdx, LUinfo,
                               policy, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(sparseHandle, trans_L, n, nnz, descr_L,
                             d_values, rowPtr, colIdx,
                             info_L, policy, pBuffer);

    cusparseDcsrsv2_analysis(sparseHandle, trans_U, n, nnz, descr_U,
                             d_values, rowPtr, colIdx,
                             info_U, policy, pBuffer);

    // step 5: A = L * U
    cusparseDcsrilu02(sparseHandle, n, nnz, descrA,
                      d_values, rowPtr, colIdx, LUinfo, policy, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &numerical_zero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseDcsrsv2_solve(sparseHandle, trans_L, n, nnz, &alpha, descr_L,
                          d_values, rowPtr, colIdx, info_L,
                          rhs, temp_solution, policy, pBuffer);
    //cusparseSpSV_solve();

    // step 7: solve U*y = z
    cusparseDcsrsv2_solve(sparseHandle, trans_U, n, nnz, &alpha, descr_U,
                          d_values, rowPtr, colIdx, info_U,
                          temp_solution, solution, policy, pBuffer);

    cudaMemcpy(Xcalculated, solution, n, cudaMemcpyDeviceToHost);

    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyCsrilu02Info(LUinfo);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_U);
    cusparseDestroy(sparseHandle);
}

int main(int argc, char **argv)
{
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

    // solveSystemSparse(sparse, B, X);
    solveSystemSparseIterative(sparse, B, X, 1e-5);
    printf("Sparse time is %f\n", toc(start));

    readSparseMMMatrix(filename, sparse); // overwrite factorized matrix to get original values for evaluation
    checkSolutionSparse(sparse, B, X, 0); // calculate |Ax-b|
    saveVector("Sparse.txt", B->size, X);
}