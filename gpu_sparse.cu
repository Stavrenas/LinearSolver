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
#include "cuda.h"

extern "C"
{
#include "utilities.h"
#include "read.h"
#include "mklILU.h"
#include "types.h"
}

void solveSystemSparseDirect(SparseMatrix *mat, Vector *B, double *X)
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

    cusparseHandle_t sparseHandle = NULL;
    cusparseCreate(&sparseHandle);

    cublasHandle_t blasHandle;
    cublasCreate(&blasHandle);

    double *rowPtrCopy, *colIdxCopy, *d_b, *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_b, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, X, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, B->values, n * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&rowPtrCopy, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMemcpy(rowPtrCopy, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&colIdxCopy, nnz * sizeof(int)));
    checkCudaErrors(cudaMemcpy(colIdxCopy, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, mat->values, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t descrX, descrB;

    cusparseCreateDnVec(&descrB, n, d_b, CUDA_R_64F);
    cusparseCreateDnVec(&descrX, n, d_x, CUDA_R_64F);

    double minusOne = -1.0;
    double one = 1.0;
    size_t spMvBufferSize = 0;
    void *spMvBuffer;

    checkCudaErrors(cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize));
    checkCudaErrors(cudaMalloc(&spMvBuffer, spMvBufferSize));

    // CALCULATE RESIDUAL and store it on B vector
    checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));

    // CUBLAS NORM
    double resNormm;
    cublasDnrm2(blasHandle, n, d_b, 1, &resNormm);

    printf("Norm  is %e\n", resNormm);
    //  printf("Status is %s\n",cudaGetErrorEnum(error));
}

void solveSystemSparseIterativeSingle(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];
    int maxIters = 15000;

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    float *zeros = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++)
        zeros[i] = 0.0;

    sortSparseMatrix(mat);

    for (int i = 0; i < nnz; i++)
        host_float_values[i] = mat->values[i];

    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle = NULL;
    cublasHandle_t blasHandle;
    cudaStream_t stream = NULL;
    // cusparseStatus_t status;

    cusparseCreate(&sparseHandle);
    cublasCreate(&blasHandle);
    cudaStreamCreate(&stream);

    // ALLOCATE MEMORY
    float *Xcalculated = (float *)malloc(n * sizeof(float));
    float *temp = (float *)malloc(nnz * sizeof(float));
    float *Lvalues, *Uvalues, *Avalues, *solution, *rhs, *rhsCopy, *temp_solutionX, *temp_solutionY;
    int *rowPtr, *colIdx, *rowPtrCopy, *colIdxCopy;

    checkCudaErrors(cudaMalloc((void **)&Uvalues, nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&Lvalues, nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&Avalues, nnz * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&rhs, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&rhsCopy, n * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&solution, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionX, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionY, n * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rowPtrCopy, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdxCopy, nnz * sizeof(int)));

    // COPY MATRIX A TO DEVICE MEMORY
    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rowPtrCopy, rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(colIdxCopy, colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    // COPY FLOAT MATRIX ELEMENTS
    checkCudaErrors(cudaMemcpy(Avalues, host_float_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // CPU ILU

    mklIncompleteLU(mat);

    // GPU LU
    // gpuLU(mat);

    // COPY FLOAT B ELEMENTS
    // cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++)
        host_float_values[i] = B->values[i];

    checkCudaErrors(cudaMemcpy(rhs, host_float_values, n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsCopy, rhs, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // INIT EMPTY VECTOR
    checkCudaErrors(cudaMemcpy(temp_solutionX, zeros, n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_solutionY, temp_solutionX, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // FREE HOST MEMORY
    free(zeros);

    for (int i = 0; i < nnz; i++)
        host_float_values[i] = mat->values[i];

    checkCudaErrors(cudaMemcpy(Lvalues, host_float_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Uvalues, Lvalues, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    cusparseSpMatDescr_t descrL, descrU, descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateCsr(&descrL, n, n, nnz, rowPtr, colIdx, Lvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateCsr(&descrU, n, n, nnz, rowPtr, colIdx, Uvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // printf("Set attributes..\n");

    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_FILL_MODE, (void *)&lower, sizeof(lower));
    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&unit, sizeof(unit));

    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_FILL_MODE, (void *)&upper, sizeof(upper));
    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&nonUnit, sizeof(nonUnit));

    // INITIALIZE B,X,Y VECTOR DESCRIPTORS
    cusparseDnVecDescr_t descrX, descrY, descrB;

    cusparseCreateDnVec(&descrB, n, rhs, CUDA_R_32F);
    cusparseCreateDnVec(&descrY, n, temp_solutionY, CUDA_R_32F);
    cusparseCreateDnVec(&descrX, n, temp_solutionX, CUDA_R_32F);

    // SETUP TRIANGULAR SOLVER DESCRIPTOR
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseSpSV_createDescr(&spsvDescrL);
    cusparseSpSV_createDescr(&spsvDescrU);
    float plusOne = 1.0;

    // INITIALIZE VARIABLES FOR LU SOLVE
    size_t spSvBufferSizeL, spSvBufferSizeU;
    void *spSvBufferL, *spSvBufferU;

    // printf("SpSv analysisL.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                            descrY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &spSvBufferSizeL));

    checkCudaErrors(cudaMalloc((void **)&spSvBufferL, spSvBufferSizeL));
    // printf("spSvBufferSizeL: %ld\n", spSvBufferSizeL);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                          descrY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, spSvBufferL));

    // printf("SpSv analysisU.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                            descrX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &spSvBufferSizeU));
    checkCudaErrors(cudaMalloc((void **)&spSvBufferU, spSvBufferSizeU));
    // printf("spSvBufferSizeU: %ld\n", spSvBufferSizeU);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                          descrX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, spSvBufferU));

    // printf("SpSv solve L.. \n");
    // // solve L*y = b
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                       descrY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    // printf("SpSv solve U.. \n");
    // // solve U*x = y
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                       descrX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

    // printf("enter loop\n");

    float minusOne = -1.0;
    float one = 1.0;
    size_t spMvBufferSize = 0;
    void *spMvBuffer;
    checkCudaErrors(cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize));
    checkCudaErrors(cudaMalloc(&spMvBuffer, spMvBufferSize));

    // calculate b norm
    float bNorm;
    cublasSnrm2(blasHandle, n, rhsCopy, 1, &bNorm);

    struct timeval tempTime;
    float spmvTime, solveTime;
    spmvTime = 0;
    solveTime = 0;

    for (int i = 0; i < maxIters; i++)
    {
        // CALCULATE RESIDUAL and store it on B vector
        tempTime = tic();
        checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));
        cudaDeviceSynchronize();
        spmvTime += toc(tempTime);

        // CUBLAS NORM
        float resNormm;
        cublasSnrm2(blasHandle, n, rhs, 1, &resNormm);

        if ((resNormm / bNorm) < tolerance)
        {
            printf("Iterations: %d\n", i);
            break;
        }

        tempTime = tic();
        // solve L*y = r : B contains the residual
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                           descrY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // solve U*c = y
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                           descrX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        cudaDeviceSynchronize();
        solveTime += toc(tempTime);

        // Xn+1 = Xn + Cn
        cublasSaxpy(blasHandle, n, &one, temp_solutionX, 1, solution, 1);
        checkCudaErrors(cudaMemcpy(temp_solutionX, solution, n * sizeof(float), cudaMemcpyDeviceToDevice));

        // restore B values
        checkCudaErrors(cudaMemcpy(rhs, rhsCopy, n * sizeof(float), cudaMemcpyDeviceToDevice));

        if (i % 100 == 0)
        {
            printf("i is %d ", i);
            printf("res Norm is %e, ", resNormm);
            printf("b norm is %e ", bNorm);
            printf("buff is %ld ", spMvBufferSize);
            printf("div is %e \n", resNormm / bNorm);
        }
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // TRANSFER SOLUTION TO X VECTOR
    checkCudaErrors(cudaMemcpy(host_float_values, temp_solutionX, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++)
        X[i] = host_float_values[i];

    printf("Spmv time is %f and Solve time is %f\n", spmvTime, solveTime);

    // FREE RESOURCES
    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrY);
    cusparseDestroyDnVec(descrB);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseDestroy(sparseHandle);
}

void solveSystemSparseIterativeDouble(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];
    int maxIters = 15000;

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    double *zeros = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
        zeros[i] = 0.0;

    sortSparseMatrix(mat);

    for (int i = 0; i < nnz; i++)
        host_float_values[i] = mat->values[i];

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
    double *Lvalues, *Uvalues, *Avalues, *solution, *rhs, *rhsCopy, *temp_solutionX, *temp_solutionY;
    int *rowPtr, *colIdx, *rowPtrCopy, *colIdxCopy;

    checkCudaErrors(cudaMalloc((void **)&Uvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Lvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Avalues, nnz * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rhs, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&rhsCopy, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&solution, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionX, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionY, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rowPtrCopy, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdxCopy, nnz * sizeof(int)));

    // COPY MATRIX A TO DEVICE MEMORY
    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rowPtrCopy, rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(colIdxCopy, colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    // COPY FLOAT MATRIX ELEMENTS
    checkCudaErrors(cudaMemcpy(Avalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // CPU ILU

    mklIncompleteLU(mat);

    // GPU LU
    // gpuLU(mat);

    // COPY FLOAT B ELEMENTS
    // cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemcpy(rhs, B->values, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsCopy, rhs, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // INIT EMPTY VECTOR
    checkCudaErrors(cudaMemcpy(temp_solutionX, zeros, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_solutionY, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // FREE HOST MEMORY
    free(zeros);

    checkCudaErrors(cudaMemcpy(Lvalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Uvalues, Lvalues, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    cusparseSpMatDescr_t descrL, descrU, descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrL, n, n, nnz, rowPtr, colIdx, Lvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrU, n, n, nnz, rowPtr, colIdx, Uvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // printf("Set attributes..\n");

    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_FILL_MODE, (void *)&lower, sizeof(lower));
    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&unit, sizeof(unit));

    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_FILL_MODE, (void *)&upper, sizeof(upper));
    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&nonUnit, sizeof(nonUnit));

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

    // INITIALIZE VARIABLES FOR LU SOLVE
    size_t spSvBufferSizeL, spSvBufferSizeU;
    void *spSvBufferL, *spSvBufferU;

    // printf("SpSv analysisL.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                            descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &spSvBufferSizeL));

    checkCudaErrors(cudaMalloc((void **)&spSvBufferL, spSvBufferSizeL));
    // printf("spSvBufferSizeL: %ld\n", spSvBufferSizeL);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                          descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, spSvBufferL));

    // printf("SpSv analysisU.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                            descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &spSvBufferSizeU));
    checkCudaErrors(cudaMalloc((void **)&spSvBufferU, spSvBufferSizeU));
    // printf("spSvBufferSizeU: %ld\n", spSvBufferSizeU);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                          descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, spSvBufferU));

    // printf("SpSv solve L.. \n");
    // // solve L*y = b
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                       descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    // printf("SpSv solve U.. \n");
    // // solve U*x = y
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                       descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

    // printf("enter loop\n");

    double minusOne = -1.0;
    double one = 1.0;
    size_t spMvBufferSize = 0;
    void *spMvBuffer;
    checkCudaErrors(cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize));
    checkCudaErrors(cudaMalloc(&spMvBuffer, spMvBufferSize));

    // calculate b norm
    double bNorm;
    cublasDnrm2(blasHandle, n, rhsCopy, 1, &bNorm);

    struct timeval tempTime;
    float spmvTime, solveTime;
    spmvTime = 0;
    solveTime = 0;

    for (int i = 0; i < maxIters; i++)
    {

        // CALCULATE RESIDUAL and store it on B vector
        tempTime = tic();
        checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));
        cudaDeviceSynchronize();
        spmvTime += toc(tempTime);

        // CUBLAS NORM
        double resNormm;
        cublasDnrm2(blasHandle, n, rhs, 1, &resNormm);

        if ((resNormm / bNorm) < tolerance)
        {
            printf("Iterations: %d\n", i);
            break;
        }

        tempTime = tic();
        // solve L*y = r : B contains the residual
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                           descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // solve U*c = y
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                           descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        cudaDeviceSynchronize();
        solveTime += toc(tempTime);

        // Xn+1 = Xn + Cn
        cublasDaxpy(blasHandle, n, &one, temp_solutionX, 1, solution, 1);
        checkCudaErrors(cudaMemcpy(temp_solutionX, solution, n * sizeof(double), cudaMemcpyDeviceToDevice));

        // restore B values
        checkCudaErrors(cudaMemcpy(rhs, rhsCopy, n * sizeof(double), cudaMemcpyDeviceToDevice));

        if (i % 100 == 0)
        {
            printf("i is %d ", i);
            printf("res Norm is %e, ", resNormm);
            printf("b norm is %e ", bNorm);
            printf("buff is %ld ", spMvBufferSize);
            printf("div is %e \n", resNormm / bNorm);
        }
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // TRANSFER SOLUTION TO X VECTOR
    checkCudaErrors(cudaMemcpy(X, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Spmv time is %f and Solve time is %f\n", spmvTime, solveTime);

    // FREE RESOURCES
    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrY);
    cusparseDestroyDnVec(descrB);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseDestroy(sparseHandle);
}

void solveSystemSparseIterativeGC(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];
    int maxIters = 1500;

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    double *zeros = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
        zeros[i] = 0.0;

    sortSparseMatrix(mat);

    for (int i = 0; i < nnz; i++)
        host_float_values[i] = mat->values[i];

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
    double *Lvalues, *Uvalues, *Avalues, *solution, *rhs, *rhsCopy, *temp_solutionX, *temp_solutionY;
    int *rowPtr, *colIdx, *rowPtrCopy, *colIdxCopy;

    checkCudaErrors(cudaMalloc((void **)&Uvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Lvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Avalues, nnz * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rhs, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&rhsCopy, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&solution, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionX, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionY, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rowPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rowPtrCopy, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdx, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&colIdxCopy, nnz * sizeof(int)));

    // COPY MATRIX A TO DEVICE MEMORY
    checkCudaErrors(cudaMemcpy(rowPtr, mat->row_idx, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colIdx, mat->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rowPtrCopy, rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(colIdxCopy, colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    // COPY FLOAT MATRIX ELEMENTS
    checkCudaErrors(cudaMemcpy(Avalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // CPU ILU

    mklIncompleteLU(mat);

    // GPU LU
    // gpuLU(mat);

    // COPY FLOAT B ELEMENTS
    // cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemcpy(rhs, B->values, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsCopy, rhs, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // INIT EMPTY VECTOR
    checkCudaErrors(cudaMemcpy(temp_solutionX, zeros, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_solutionY, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // FREE HOST MEMORY
    free(zeros);

    checkCudaErrors(cudaMemcpy(Lvalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Uvalues, Lvalues, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    cusparseSpMatDescr_t descrL, descrU, descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrL, n, n, nnz, rowPtr, colIdx, Lvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrU, n, n, nnz, rowPtr, colIdx, Uvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // printf("Set attributes..\n");

    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t nonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_FILL_MODE, (void *)&lower, sizeof(lower));
    cusparseSpMatSetAttribute(descrL, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&unit, sizeof(unit));

    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_FILL_MODE, (void *)&upper, sizeof(upper));
    cusparseSpMatSetAttribute(descrU, CUSPARSE_SPMAT_DIAG_TYPE, (void *)&nonUnit, sizeof(nonUnit));

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

    // INITIALIZE VARIABLES FOR LU SOLVE
    size_t spSvBufferSizeL, spSvBufferSizeU;
    void *spSvBufferL, *spSvBufferU;

    // printf("SpSv analysisL.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                            descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &spSvBufferSizeL));

    checkCudaErrors(cudaMalloc((void **)&spSvBufferL, spSvBufferSizeL));
    // printf("spSvBufferSizeL: %ld\n", spSvBufferSizeL);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                          descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, spSvBufferL));

    // printf("SpSv analysisU.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                            descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &spSvBufferSizeU));
    checkCudaErrors(cudaMalloc((void **)&spSvBufferU, spSvBufferSizeU));
    // printf("spSvBufferSizeU: %ld\n", spSvBufferSizeU);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                          descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, spSvBufferU));

    // printf("SpSv solve L.. \n");
    // // solve L*y = b
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                       descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    // printf("SpSv solve U.. \n");
    // // solve U*x = y
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                       descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

    // printf("enter loop\n");

    double minusOne = -1.0;
    double one = 1.0;
    double zero = 0.0;
    size_t spMvBufferSize = 0;
    void *spMvBuffer;
    checkCudaErrors(cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize));
    checkCudaErrors(cudaMalloc(&spMvBuffer, spMvBufferSize));

    // calculate b norm
    double bNorm;
    cublasDnrm2(blasHandle, n, rhsCopy, 1, &bNorm);

    // for conjucate gradient
    double *P, *q, *z;
    cudaMalloc((void **)&P, n * sizeof(double));
    cudaMalloc((void **)&q, n * sizeof(double));
    cudaMalloc((void **)&z, n * sizeof(double));

    cusparseDnVecDescr_t descrP, descrQ, descrZ;

    cusparseCreateDnVec(&descrP, n, P, CUDA_R_64F);
    cusparseCreateDnVec(&descrQ, n, q, CUDA_R_64F);
    cusparseCreateDnVec(&descrZ, n, z, CUDA_R_64F);

    double Pi = 0.0;
    double Pi_prev = 0.0;
    double beta, alpha;
    double resNormm, qNorm, pNorm, zNorm;

    struct timeval tempTime;
    float spmvTime, solveTime;
    spmvTime = 0;
    solveTime = 0;

    for (int i = 0; i < maxIters; i++)
    {

        // checkCudaErrors(cublasDswap(blasHandle, n, rhs, 1, rhsCopy, 1)); // swap vectors to calculate residual for convergence (rhs is corelated to descrB)
        // // CALCULATE RESIDUAL and store it on B vector
        // checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));

        // checkCudaErrors(cublasDswap(blasHandle, n, rhs, 1, rhsCopy, 1)); // restore rhs values (r in GC algorithm)

        // Step 3: solve Az <- r //
        // solve L*y = r : B contains the residual
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                           descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // solve U*z = y
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                           descrZ, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));
        // z is stored on descrz

        // cublasDnrm2(blasHandle, n, z, 1, &zNorm);
        // printf("z norm is %e ", zNorm);

        // Step 4: pi = r^T * z //

        Pi_prev = Pi;
        checkCudaErrors(cublasDdot(blasHandle, n, rhs, 1, z, 1, &Pi));
        // printf("Pi is %e ", Pi);

        // Step 5-6: if i == 0 , p <-z //
        if (i == 0)
            cublasDcopy(blasHandle, n, z, 1, P, 1);

        else
        {
            // Step 7-8: beta <- pi/pi_1 //
            beta = Pi / Pi_prev;
            // printf("beta is %e ", beta);

            // Step 9: z <- z + bp
            cublasDaxpy(blasHandle, n, &beta, P, 1, z, 1); // result is saved on z
            //  p <-z //
            cublasDcopy(blasHandle, n, z, 1, P, 1);
        }

        // cublasDnrm2(blasHandle, n, P, 1, &pNorm);
        // printf("p norm is %e ", pNorm);

        // Step 10-11: compute q <- Ap
        checkCudaErrors(cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, descrACopy, descrP, &zero, descrQ, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer));
        // cublasDnrm2(blasHandle, n, q, 1, &qNorm);
        // printf("q norm is %e ", qNorm);

        // Step 12: a <- Pi / P^T * q
        double temp;
        checkCudaErrors(cublasDdot(blasHandle, n, P, 1, q, 1, &temp));
        alpha = Pi / temp;
        // printf("a is %e ", alpha);

        // Step 13: Xn+1 = Xn + a * p
        checkCudaErrors(cublasDaxpy(blasHandle, n, &alpha, P, 1, temp_solutionX, 1));

        alpha *= -1;
        // Step 14: r <- r - a * q
        cublasDaxpy(blasHandle, n, &alpha, q, 1, rhs, 1); // result is saved on rhs

        // cublasDnrm2(blasHandle, n, rhs, 1, &resNormm);
        // printf("res Norm is %e, ", resNormm);

        // RESIDUAL NORM
        cublasDnrm2(blasHandle, n, rhs, 1, &resNormm);

        if ((resNormm / bNorm) < tolerance)
        {
            printf("Iterations: %d\n", i);
            break;
        }

        if (i % 100 == 0)
        {
            printf("i is %d ", i);
            printf("res Norm is %e, ", resNormm);
            printf("a is %e  beta is %e  ", alpha, beta);
            printf("div is %e \n", resNormm / bNorm);
        }
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // TRANSFER SOLUTION TO X VECTOR
    checkCudaErrors(cudaMemcpy(X, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToHost));

    // printf("Spmv time is %f and Solve time is %f\n", spmvTime, solveTime);

    // FREE RESOURCES
    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrY);
    cusparseDestroyDnVec(descrB);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseDestroy(sparseHandle);
}

int main(int argc, char **argv)
{
    char *matrixName = (char *)malloc(40 * sizeof(char));
    char *temp = (char *)"data/n10k.bin";
    char saveFile[40] = "var/GPUX.txt";

    if (argc == 2)
        strcpy(matrixName, argv[1]);
    else
        strcpy(matrixName, temp);

    SparseMatrix *sparse = (SparseMatrix *)malloc(sizeof(SparseMatrix));
    Vector *B = (Vector *)malloc(sizeof(Vector));
    Vector *Xcorrect = (Vector *)malloc(sizeof(Vector));

    if (strstr(matrixName, ".bin"))
        readSystem(matrixName, sparse, B, Xcorrect);

    else if (strstr(matrixName, ".mtx"))
    {
        char *filenameB = (char *)malloc(40 * sizeof(char));
        char name[40];

        strcpy(name, matrixName);
        name[strlen(name) - 4] = (char)'\0';
        sprintf(filenameB, "%s_rhs1.mtx", name);
        readSparseMMMatrix(matrixName, sparse);
        readMMVector(filenameB, B);
    }

    double *X = (double *)malloc(B->size * sizeof(double));

    struct timeval start = tic();

    for (int i = 0; i < 1; i++)
        solveSystemSparseIterativeGC(sparse, B, X, 1e-7);
    // solveSystemSparseDirect(sparse, B, X);

    printf("Sparse time is %f\n", toc(start));

    saveVector(saveFile, B->size, X);
}
