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
#include "cudaUtilities.h"
#include "helper_cuda.h"

extern "C"
{
#include "utilities.h"
#include "read.h"
#include "mklILU.h"
#include "types.h"
}

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

    // INITIALIZE CUSOLVER
    cusparseHandle_t sparseHandle = NULL;
    cusparseCreate(&sparseHandle);

    size_t bufferSize;
    void *buffer;
    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(sparseHandle, n, n, nnz, values, rowPtr, colIdx, values,
                                                  rowPtr, colIdx, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                                  CUSPARSE_CSR2CSC_ALG1, &bufferSize));

    printf("Buffer size is %d\n", bufferSize);
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    checkCudaErrors(cusparseCsr2cscEx2(sparseHandle, n, n, nnz, values, rowPtr, colIdx, values,
                                       rowPtr, colIdx, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                       CUSPARSE_CSR2CSC_ALG1, buffer));

    // RUN TWICE TO GET SORTED MATRIX
    checkCudaErrors(cusparseCsr2cscEx2(sparseHandle, n, n, nnz, values, rowPtr, colIdx, values,
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
    cudaFree(buffer);
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
    int maxIters = 5000;

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    double *zeros = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
    {
        zeros[i] = 0.0;
    }

    for (int i = 0; i < nnz; i++)
    {
        host_float_values[i] = mat->values[i];
    }

    sortSparseMatrix(mat);

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
    double *temp = (double *)malloc(nnz * sizeof(double));
    double *Lvalues, *Uvalues, *Avalues, *solution, *rhs, *rhsCopy, *temp_solutionX, *temp_solutionY;
    float *f_values;
    int *rowPtr, *colIdx, *rowPtrCopy, *colIdxCopy;

    checkCudaErrors(cudaMalloc((void **)&Uvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Lvalues, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&Avalues, nnz * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&rhs, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&rhsCopy, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&solution, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionX, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&temp_solutionY, n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&f_values, nnz * sizeof(float)));

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
    checkCudaErrors(cudaMemcpy(f_values, host_float_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Avalues, mat->values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // CPU ILU
    printSparseMatrix(mat);
    mklIncompleteLU(mat);
    printSparseMatrix(mat);

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

    printf("\ndone converting..\n");

    cusparseSpMatDescr_t descrL, descrU, descrACopy;
    // Create a copy of A to calculate residual r = b - Ax
    cusparseCreateCsr(&descrACopy, n, n, nnz, rowPtrCopy, colIdxCopy, Avalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrL, n, n, nnz, rowPtr, colIdx, Lvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&descrU, n, n, nnz, rowPtr, colIdx, Uvalues, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    printf("Set attributes..\n");

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

    printf("SpSv analysisL.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                            descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &spSvBufferSizeL));

    cudaMalloc((void **)&spSvBufferL, spSvBufferSizeL);
    printf("spSvBufferSizeL: %ld\n", spSvBufferSizeL);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                          descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, spSvBufferL));

    printf("SpSv analysisU.. \n");
    checkCudaErrors(cusparseSpSV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                            descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &spSvBufferSizeU));
    cudaMalloc((void **)&spSvBufferU, spSvBufferSizeU);
    printf("spSvBufferSizeU: %ld\n", spSvBufferSizeU);

    checkCudaErrors(cusparseSpSV_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                          descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, spSvBufferU));

    printf("SpSv solve L.. \n");
    // solve L*y = b
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                       descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    printf("SpSv solve U.. \n");
    // solve U*x = y
    checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                       descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

    // cudaMemcpy(X, temp_solutionY, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    // for (int j = 0; j < 3; j++)
    //     printf("%e ", X[j]);
    // printf("\n");

    printf("enter loop\n");

    for (int i = 0; i < maxIters; i++)
    {

        double minusOne = -1.0;
        double one = 1.0;
        size_t spMvBufferSize = 0;
        void *spMvBuffer;

        // CALCULATE RESIDUAL and store it on B vector
        cusparseSpMV_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &spMvBufferSize);
        cudaMalloc(&spMvBuffer, spMvBufferSize);
        cusparseSpMV(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minusOne, descrACopy, descrX, &one, descrB, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, spMvBuffer);

        // CUBLAS NORM
        double resNormm, bNorm;
        cublasDnrm2(blasHandle, n, rhs, 1, &resNormm);
        cublasDnrm2(blasHandle, n, rhsCopy, 1, &bNorm);

        if ((resNormm / bNorm) < tolerance)
        {
            printf("Iters: %d\n", i);
            break;
        }
        // solve L*y = r : B contains the residual
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrL, descrB,
                                           descrY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // solve U*c = y
        checkCudaErrors(cusparseSpSV_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &plusOne, descrU, descrY,
                                           descrX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));
        // Xn+1 = Xn + Cn
        cublasDaxpy(blasHandle, n, &one, temp_solutionX, 1, solution, 1);
        cudaMemcpy(temp_solutionX, solution, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // restore B values
        cudaMemcpy(rhs, rhsCopy, n * sizeof(double), cudaMemcpyDeviceToDevice);

        if (i % 100 == 0)
        {
            printf("i is %d ", i);
            printf("res Norm is %e, ", resNormm);
            printf("b norm is %f ", bNorm);
            printf("buff is %ld ", spMvBufferSize);
            printf("div is %e \n", resNormm / bNorm);
            // cudaMemcpy(X, temp_solutionX, n * sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    // TRANSFER SOLUTION TO X VECTOR
    cudaMemcpy(X, temp_solutionX, n * sizeof(double), cudaMemcpyHostToDevice);

    cusparseDestroyDnVec(descrX);
    cusparseDestroyDnVec(descrY);
    cusparseDestroyDnVec(descrB);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseDestroy(sparseHandle);

    cudaMemcpy(X, solution, n * sizeof(double), cudaMemcpyDeviceToHost);

    // FREE RESOURCES
}

// USED TO READ .mtx FILES AND THE CORRESPONDING rhs.mtx
void solveMtx(int argc, char **argv)
{
    const char *matrixName;
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

    for (int i = 0; i < 1; i++)
        // solveSystemSparseIterative(sparse, B, X, 1e-12);
        mklIncompleteLU(sparse);

    printf("Gpu time is %f\n", toc(start));

    // saveVector("var/Sparse.txt", B->size, X);
}

// USED TO READ .bin FILES WHICH INCLUDE THE RIGHT HAND SIDE AND THE SOLUTION
void solveBin(int argc, char **argv)
{
    char *matrixName;

    if (argc == 2)
        matrixName = argv[1];
    else
        matrixName = "data/n10k.bin";
    SparseMatrix *sparse = (SparseMatrix *)malloc(sizeof(SparseMatrix));
    Vector *B = (Vector *)malloc(sizeof(Vector));
    Vector *Xcorrect = (Vector *)malloc(sizeof(Vector));
    double *X = (double *)malloc(sizeof(double));
    readSystem(matrixName, sparse, B, Xcorrect);

    // printSparseMatrix(sparse);

    struct timeval start = tic();

    for (int i = 0; i < 1; i++)
        solveSystemSparseIterative(sparse, B, X, 1e-12);
    // mklIncompleteLU(sparse);

    printf("Sparse time is %f\n", toc(start));

    saveVector("var/X.txt", B->size, X);
}

int main(int argc, char **argv)
{
    // solveMtx(argc, argv);
    solveBin(argc, argv);
}
