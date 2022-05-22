#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverSp.h>
#include <cusolverRf.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cudaUtilities.h"
#include "helper_cuda.h"

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
        cusolverHandle, n, nnz, descrA,
        mat->values,
        mat->row_idx,
        mat->col_idx,
        B->values,
        tol, // tolerance to determine matrix singularity
        reorder, X, &singularity);

    cusolverSpDestroy(cusolverHandle);

    printf("Singularity is %d\n", singularity);
    // printf("Status is %s\n",cudaGetErrorEnum(error));
}

void solveSystemSparseIterative(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];
    double tol = 1e-8;
    int maxIters = 5;
    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    float *host_float_rhs = (float *)malloc(n * sizeof(float));
    double *zeros = (double *)malloc(n * sizeof(double));

    int maxThreads, blocks;
    if (nnz > 512)
    {
        maxThreads = 512;
        blocks = nnz / maxThreads + 1;
    }
    else
    {
        blocks = 1;
        maxThreads = nnz;
    }
    for (int i = 0; i < n; i++)
    {
        host_float_rhs[i] = B->values[i];
        zeros[i] = 0.0;
    }

    for (int i = 0; i < nnz; i++)
    {
        host_float_values[i] = mat->values[i];
    }

    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle = NULL;
    cublasHandle_t blasHandle;
    cudaStream_t stream = NULL;
    // cusparseStatus_t status;

    cusparseCreate(&sparseHandle);
    cublasCreate(&blasHandle);
    cudaStreamCreate(&stream);

    // ALLOCATE MEMORY
    double *Xcalculated = (double *)malloc(n * sizeof(double));
    float *Xcalculatedf = (float *)malloc(n * sizeof(float));
    double *Lvalues, *Uvalues, *Avalues, *solution, *rhs, *rhsCopy, *temp_solutionX, *temp_solutionY;
    float *f_values, *tempBig, *temp;
    int *rowPtr, *colIdx, *rowPtrCopy, *colIdxCopy;

    cudaMalloc((void **)&Lvalues, nnz * sizeof(double));
    cudaMalloc((void **)&Uvalues, nnz * sizeof(double));
    cudaMalloc((void **)&Avalues, nnz * sizeof(double));
    cudaMalloc((void **)&solution, n * sizeof(double));
    cudaMalloc((void **)&rhs, n * sizeof(double));
    cudaMalloc((void **)&rhsCopy, n * sizeof(double));
    cudaMalloc((void **)&temp_solutionX, n * sizeof(double));
    cudaMalloc((void **)&temp_solutionY, n * sizeof(double));

    cudaMalloc((void **)&temp, n * sizeof(float));
    cudaMalloc((void **)&f_values, nnz * sizeof(float));
    cudaMalloc((void **)&tempBig, nnz * sizeof(float));

    cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int));
    cudaMalloc((void **)&rowPtrCopy, (n + 1) * sizeof(int));
    cudaMalloc((void **)&colIdx, nnz * sizeof(int));
    cudaMalloc((void **)&colIdxCopy, nnz * sizeof(int));

    // COPY MATRIX A TO DEVICE MEMORY
    cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rowPtrCopy, rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(colIdxCopy, colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice);

    // COPY FLOAT MATRIX ELEMENTS
    cudaMemcpy(f_values, host_float_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Avalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // COPY FLOAT B ELEMENTS
    // cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs, B->values, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rhsCopy, rhs, n * sizeof(double), cudaMemcpyDeviceToDevice);

    // INIT EMPTY VECTOR
    cudaMemcpy(temp_solutionX, zeros, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_solutionY, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToDevice);

    // FREE HOST MEMORY
    free(host_float_rhs);
    free(zeros);
    // free(host_float_values);

    // SETUP MATRIX DESCRIPTOR
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // INITIALIZE VARIABLES FOR LU FACTORIZATION
    int pBufferSize;
    size_t spSvBufferSizeL, spSvBufferSizeU;
    void *pBuffer, *spSvBufferL, *spSvBufferU;
    // int structural_zero, numerical_zero;

    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    csrilu02Info_t LUinfo;
    cusparseCreateCsrilu02Info(&LUinfo);

    double tole = 0;
    float boost = 1e-8;
    cusparseScsrilu02_numericBoost(sparseHandle, LUinfo, 1, &tole, &boost);

    // CALCULATE LU FACTORIZATION BUFFER SIZE
    checkCudaErrors(cusparseScsrilu02_bufferSize(sparseHandle, n, nnz, descrA,
                                                 f_values, rowPtr, colIdx, LUinfo, &pBufferSize));

    cudaMalloc(&pBuffer, pBufferSize);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes
    // printf("Buffer size for LU is %d\n",pBufferSize);

    // LU FACTORIZATION ANALYSIS
    checkCudaErrors(cusparseScsrilu02_analysis(sparseHandle, n, nnz, descrA,
                                               f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    cusparseStatus_t status;
    int structural_zero;
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);

    // A = L * U
    checkCudaErrors(cusparseScsrilu02(sparseHandle, n, nnz, descrA,
                                      f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    // f_values now contain L U matrices

    cusparseDestroyMatDescr(descrA);
    cudaFree(pBuffer);
    cusparseDestroyCsrilu02Info(LUinfo);

    floatToDoubleVector<<<blocks, maxThreads>>>(f_values, Lvalues);
    cudaMemcpy(Uvalues, Lvalues, nnz * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaFree(f_values);

    cusparseSpMatDescr_t descrL, descrU, descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrL, n, n, nnz, rowPtr, colIdx, Lvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrU, n, n, nnz, rowPtr, colIdx, Uvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_FILL_MODE, (void *)&lower, sizeof(lower));
    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&unit, sizeof(unit));

    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_FILL_MODE, (void *)&upper, sizeof(upper));
    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&nonUnit, sizeof(nonUnit));

    // cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)); // THIS WORKS
    // cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof(unit));

    // INITIALIZE B,X,Y VECTOR DESCRIPTORS
    cusparseDnVecDescr_t descrX, descrY, descrB;

    cusparseCreateDnVec(&descrB, n, rhs, CUDA_R_64F);
    cusparseCreateDnVec(&descrY, n, temp_solutionY, CUDA_R_64F);
    cusparseCreateDnVec(&descrX, n, temp_solutionX, CUDA_R_64F);

    // SETUP TRIANGULAR SOLVER DESCRIPTOR
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseSpSV_createDescr(&spsvDescrL);
    cusparseSpSV_createDescr(&spsvDescrU);
    double plusOne = 1.0;

    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                            descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &spSvBufferSizeL));
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                            descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &spSvBufferSizeU));

    cudaMalloc((void **)&spSvBufferU, spSvBufferSizeU);
    cudaMalloc((void **)&spSvBufferL, spSvBufferSizeL);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                          descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, spSvBufferL));

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                          descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, spSvBufferU));

    // solve L*y = b
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                       descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    // solve U*x = y
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                       descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

    // cudaMemcpy(X, temp_solutionY, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    // for (int j = 0; j < 3; j++)
    //     printf("%e ", X[j]);
    // printf("\n");

    cudaMemcpy(X, temp_solutionX, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < 3; j++)
        printf("%e ", X[j]);
    printf("\n");

    // cudaMemcpy(X, Uvalues, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    // for (int j = 0; j < 5; j++)
    //     printf("%e ", X[j]);
    // printf("\n");

    for (int i = 0; i < maxIters; i++)
    {
        // CALCULATE RESIDUAL and store it on B vector
        double minusOne = -1.0;
        double one = 1.0;
        size_t spMvBufferSize = 0;
        ;
        void *spMvBuffer;

        cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize);
        cudaMalloc(&spMvBuffer, spMvBufferSize);
        cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer);

        printf("Res is ");
        cudaMemcpy(X, rhs, 3 * sizeof(double), cudaMemcpyDeviceToHost);
        for (int j = 0; j < 3; j++)
            printf("%e ", X[j]);
        printf("\n");

        // CUBLAS NORM
        double resNormm, bNorm;
        cublasDnrm2(blasHandle, n, rhs, 1, &resNormm);
        cublasDnrm2(blasHandle, n, rhsCopy, 1, &bNorm);
        printf("res Norm is %f, b norm is %f and div is %f , buff is %d  \n", resNormm, bNorm, resNormm / bNorm, spMvBufferSize);

        if ((resNormm / bNorm) < tol)
            break;

        // Xn+1 = Xn + Cn
        cublasDaxpy(blasHandle, n, &one, temp_solutionX, 1, solution, 1);
        cudaMemcpy(temp_solutionX, solution, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // solve L*y = r
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                           descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        //  step 7: solve U*c = y
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                           descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        // restore B values
        cudaMemcpy(rhs, rhsCopy, n * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(X, rhs, n * sizeof(double), cudaMemcpyHostToDevice);
    // for (int j = 0; j < 3; j++)
    //     printf("%e ", X[j]);
    // printf("\n");

    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrY);
    cusparseDestroyDnVec(descrB);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseDestroy(sparseHandle);

    cudaMemcpy(X, solution, n * sizeof(double), cudaMemcpyDeviceToHost);

    // FREE RESOURCES
}

int main(int argc, char **argv)
{
    char *matrixName;
    if (argc == 1)
        matrixName = "data/sherman1";
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