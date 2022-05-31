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
#include "cudaUtilities.h"
#include "helper_cuda.h"

extern "C"
{
#include "utilities.h"
#include "read.h"
}

// Calculate product aik * akj. rowStart and rowEnd is for row i
bool rowColProduct(int rowStarti, int rowEndi, int rowStartk, int rowEndk, int k, int j, SparseMatrix *mat, float *result)
{
    int size = mat->size;

    // search for aik
    for (int aik = rowStarti; aik < rowEndi; aik++)
    {
        if (mat->col_idx[aik] != k)
            continue;

        // search for akj
        for (int akj = rowStartk; akj < rowEndk; akj++)
        {
            if (mat->col_idx[akj] != j)
                continue;

            *result = mat->values[aik] * mat->values[akj];
            // printf("a(x,%d) * a(%d,%d)\n", k, k, j);
            return true;
        }
    }
    return false;
}

void incompleteLU(SparseMatrix *mat)
{

    int size = mat->size;
    int nnz = mat->row_idx[size];

    float *fvaluesLU = (float *)malloc(nnz * sizeof(float));
    double *temp = (double *)malloc(nnz * sizeof(double));

    for (int i = 0; i < nnz; i++)
    {
        temp[i] = 0;
        fvaluesLU[i] = mat->values[i];
    }

    // iterate through rows
    for (int i = 1; i < size; i++)
    {

        // calculate L
        for (int k = 0; k < i; k++)
        {

            // search for akk
            float akk = 0.0;
            int startk = mat->row_idx[k];
            int endk = mat->row_idx[k + 1];
            bool found = false;

            for (int indexk = startk; indexk < endk; indexk++)
            {
                if (mat->col_idx[indexk] != k)
                    continue;

                akk = fvaluesLU[indexk];
                found = true;
                break;
            }

            if (!found)
                exit(-100);

            // Compute aik = aik/akk
   
            int starti = mat->row_idx[i];
            int endi = mat->row_idx[i + 1];
            for (int j = starti; j < endi; j++)
            {
                if (mat->col_idx[j] != k)
                    continue;

                //printf("a%d%d /= a%d%d\n", i, k, k, k);
                fvaluesLU[j] = fvaluesLU[j] / akk;
                break;
            }

            for (int j = k + 1; j < mat->size; j++)
            {
                int start = mat->row_idx[j];
                int end = mat->row_idx[j + 1];

                // Compute aij -= - aik * akj
                for (int indexj = starti; indexj < endi; indexj++)
                {
                    if (mat->col_idx[indexj] != j)
                        continue;

                    float result = 0.0;
                     //printf("a%d%d -=  a%d%d / a%d%d\n", i, j, i, k, k, j);
                    // calculate product aik * akj

                    if (rowColProduct(starti, endi, startk, endk, k, j, mat, &result))
                        fvaluesLU[indexj] = fvaluesLU[indexj] - result;
                }
            }
        }
    }

    for (int i = 0; i < nnz; i++)
        temp[i] = fvaluesLU[i];

    free(fvaluesLU);
    mat->values = temp;
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
    float *host_float_rhs = (float *)malloc(n * sizeof(float));
    double *zeros = (double *)malloc(n * sizeof(double));

    float *tempf = (float *)malloc(nnz * sizeof(float));
    double *tempd = (double *)malloc(nnz * sizeof(double));

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

    // COPY FLOAT B ELEMENTS
    // cudaMemcpy(rhs, B->values, n, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemcpy(rhs, B->values, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsCopy, rhs, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // INIT EMPTY VECTOR
    checkCudaErrors(cudaMemcpy(temp_solutionX, zeros, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp_solutionY, temp_solutionX, n * sizeof(double), cudaMemcpyDeviceToDevice));

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
    checkCudaErrors(cusparseScsrilu02_numericBoost(sparseHandle, LUinfo, 1, &tole, &boost));

    printf("Buffer size..\n");
    // CALCULATE LU FACTORIZATION BUFFER SIZE

    checkCudaErrors(cusparseScsrilu02_bufferSize(sparseHandle, n, nnz, descrA,
                                                 f_values, rowPtr, colIdx, LUinfo, &pBufferSize));

    checkCudaErrors(cudaMalloc(&pBuffer, pBufferSize));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes
    printf("Buffer size for LU is %d\n", pBufferSize);

    printf("Analysis..\n");
    // LU FACTORIZATION ANALYSIS
    checkCudaErrors(cusparseScsrilu02_analysis(sparseHandle, n, nnz, descrA,
                                               f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    cusparseStatus_t status;
    int structural_zero;
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);

    printf("Factorization..\n");
    // A = L * U
    checkCudaErrors(cusparseScsrilu02(sparseHandle, n, nnz, descrA,
                                      f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));
    cudaDeviceSynchronize(); // this stalls

    // f_values now contain L U matrices
    //  cusparseDestroyMatDescr(descrA);
    //  cudaFree(pBuffer);
    //  cusparseDestroyCsrilu02Info(LUinfo);
    cudaError_t err;
    printf("Convert to double..\n");

    // size_t free, total;
    // cudaMemGetInfo(&free,&total);
    // printf("Free is %ld and total is %ld\n", free, total);

    // DEVICE TYPECAST
    // floatToDoubleVector<<<blocks, maxThreads>>>(f_values, Lvalues, nnz);

    // OR HOST TYPECAST
    cudaMemcpy(host_float_values, f_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nnz; i++)
        tempd[i] = host_float_values[i];

    cudaMemcpy(Lvalues, tempd, nnz * sizeof(double), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMemcpy(Uvalues, Lvalues, nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(temp, Lvalues, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tempf, f_values, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    // END HOST TYPECAST

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

void LU(SparseMatrix *mat)
{

    int n = mat->size;
    int nnz = mat->row_idx[n];

    // create float copy of system elements
    float *host_float_values = (float *)malloc(nnz * sizeof(float));
    float *host_float_valuesLU = (float *)malloc(nnz * sizeof(float));

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
    float *f_values;
    int *rowPtr, *colIdx;

    checkCudaErrors(cudaMalloc((void **)&f_values, nnz * sizeof(float)));

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
    size_t spSvBufferSizeL, spSvBufferSizeU;
    void *pBuffer, *spSvBufferL, *spSvBufferU;
    // int structural_zero, numerical_zero;

    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    csrilu02Info_t LUinfo;
    cusparseCreateCsrilu02Info(&LUinfo);

    double tole = 0;
    float boost = 1e-8;
    checkCudaErrors(cusparseScsrilu02_numericBoost(sparseHandle, LUinfo, 1, &tole, &boost));

    printf("Buffer size..\n");
    // CALCULATE LU FACTORIZATION BUFFER SIZE

    checkCudaErrors(cusparseScsrilu02_bufferSize(sparseHandle, n, nnz, descrA,
                                                 f_values, rowPtr, colIdx, LUinfo, &pBufferSize));

    checkCudaErrors(cudaMalloc(&pBuffer, pBufferSize));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes
    printf("Buffer size for LU is %d\n", pBufferSize);

    printf("Analysis..\n");
    // LU FACTORIZATION ANALYSIS
    checkCudaErrors(cusparseScsrilu02_analysis(sparseHandle, n, nnz, descrA,
                                               f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));

    cusparseStatus_t status;
    int structural_zero;
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, LUinfo, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);

    printf("Factorization..\n");
    // A = L * U
    checkCudaErrors(cusparseScsrilu02(sparseHandle, n, nnz, descrA,
                                      f_values, rowPtr, colIdx, LUinfo, policy, pBuffer));
    cudaMemcpy(host_float_valuesLU, f_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    double *temp = (double *)malloc(nnz * sizeof(double));
    for (int i = 0; i < nnz; i++)
        temp[i] = host_float_valuesLU[i];
    mat->values = temp;

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
        LU(sparse);
    printf("Gpu time is %f\n", toc(start));
    start = tic();

    incompleteLU(sparse);
    printf("Cpu time is %f\n", toc(start));

    saveVector("Sparse.txt", B->size, X);
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

    printSparseMatrix(sparse);

    struct timeval start = tic();

    for (int i = 0; i < 1; i++)
        //solveSystemSparseIterative(sparse, B, X, 1e-8);
    incompleteLU(sparse) ;
    printf("Sparse time is %f\n", toc(start));

    saveVector("X.txt", B->size, X);
}

int main(int argc, char **argv)
{
    solveMtx(argc, argv);
   //solveBin(argc, argv);
}
