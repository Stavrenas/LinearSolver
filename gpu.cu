#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"

extern "C"
{
#include "utilities.h"
#include "read.h"
#include "cudaUtilities.h"
}


void iterativeRefinementGeneral(DenseMatrix *A, Vector *B, double *X)
{

    // INITIALIZE CUSOLVER
    cusolverDnHandle_t cusolverHandle;
    cudaStream_t stream = NULL;
    cusolverStatus_t error;
    cusolverDnIRSParams_t params;
    cusolverDnIRSInfos_t infos;

    cusolverDnCreate(&cusolverHandle);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverHandle, stream);
    cusolverDnIRSInfosCreate(&infos);

    /* SET ITERATIVE REFINEMENT PARAMETERS*/
    cusolverDnIRSParamsCreate(&params);
    cusolverDnIRSParamsSetSolverPrecisions(params, CUSOLVER_R_64F, CUSOLVER_R_32F); // main and lowest solver precision
    cusolverDnIRSParamsSetRefinementSolver(params, CUSOLVER_IRS_REFINE_CLASSICAL);
    cusolverDnIRSParamsSetTol(params, 1e-12);
    /*This function sets the tolerance for the refinement solver. By default it is such that all the RHS satisfy:

        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX where
    RNRM is the infinity-norm of the residual
    XNRM is the infinity-norm of the solution
    ANRM is the infinity-operator-norm of the matrix A
    EPS is the machine epsilon for the Inputs/Outputs datatype that matches LAPACK <X>LAMCH('Epsilon')
    BWDMAX, the value BWDMAX is fixed to 1.0
    */


    cusolverDnIRSParamsSetTolInner(params, 1e-4);    // default value is 1e-4
    cusolverDnIRSParamsSetMaxIters(params, 50);      // default value is 50
    cusolverDnIRSParamsSetMaxItersInner(params, 50); // default value is 50
    cusolverDnIRSParamsDisableFallback(params); //by default enabled

    /* SET ITERATIVE REFINEMENT INFOS*/
    cusolver_int_t max_iters, n_iters, outer_iters;
    cusolverDnIRSInfosGetMaxIters(infos, &max_iters);

    int info, sides, size;
    int *d_info = nullptr; /* device error info */
    int *iters = (int *)malloc(sizeof(int));

    size_t Lwork = 0;          /* size of workspace */
    void *Workspace = nullptr; /* device workspace */

    /* step 2: COPY MATRICES TO DEVICE */
    double *Bcuda, *Acuda, *Xcuda;
    size = A->size;

    cudaMalloc((void **)&Acuda, size * size * sizeof(double));
    cudaMalloc((void **)&Bcuda, size * sizeof(double));
    cudaMalloc((void **)&Xcuda, size * sizeof(double));

    cudaMalloc((void **)&d_info, sizeof(int));

    cudaMemcpy(Acuda, A->values, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Bcuda, B->values, size * sizeof(double), cudaMemcpyHostToDevice);

    cusolver_int_t n = size;
    cusolver_int_t nrhs = 1;
    sides = 1;
    /* step 3: query working space */
    error = cusolverDnIRSXgesv_bufferSize(cusolverHandle, params, n, nrhs, &Lwork);
    cudaMalloc(&Workspace, Lwork);
    // printf("Error: %s\n", cudaGetErrorEnum(error));

    /* step 4: Iterative Refinement solution */
    error = cusolverDnIRSXgesv(cusolverHandle, params, infos, size, sides, Acuda, size, Bcuda, size, Xcuda, size, Workspace, Lwork, iters, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Error: %s\n", cudaGetErrorEnum(error));

    if (0 != info)
    {
        if (info < 0)
            printf("%d-th parameter is wrong \n", -info);
        else
            printf("U(%d,%d) is exactly zero \n", info, info);
        exit(1);
    }

    //cusolverDnIRSInfosGetOuterNiters(infos, &outer_iters);
    cusolverDnIRSInfosGetNiters(infos, &n_iters);

    printf("Iters: %d\n", n_iters);
    printf("Size of Workspace: %ld\n", Lwork);

    cudaMemcpy(X, Xcuda, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* free resources */
    cudaFree(Acuda);
    cudaFree(Bcuda);
    cudaFree(Xcuda);
    cudaFree(d_info);
    cudaFree(Workspace);

    // cublasDestroy(handle);
    cusolverDnDestroy(cusolverHandle);
    cudaDeviceReset();
}

int main(int argc, char *argv[])
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
    DenseMatrix *dense = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    Vector *B = (Vector *)malloc(sizeof(Vector));

    readSparseMMMatrix(filename, sparse);
    readMMVector(filenameB, B);
    sparseToDense(sparse, dense);

    double *X = (double *)malloc(B->size * sizeof(double));

    struct timeval start = tic();

    for (int i = 0; i < 1; i++)
        iterativeRefinementGeneral(dense, B, X);

    printf("Iterative refinement time is %f\n", toc(start));

    readSparseMMMatrix(filename, sparse);
    sparseToDense(sparse, dense);    // overwrite factorized matrix to get original values for evaluation
    checkSolutionDense(dense, B, X, 0); // calculate |Ax-b|
    saveVector("Dense.txt",B->size,X);
}
