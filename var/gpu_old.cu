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

void Cholesky(double **A, double *B, double *X, int size)
{

    double *Aserialized = (double *)malloc(size * size * sizeof(double));
    serializeMatrix(size, A, Aserialized);

    // INITIALIZE CUSOLVER
    cusolverDnHandle_t cusolverHandle;
    cudaStream_t stream = NULL;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    cusolverDnCreate(&cusolverHandle);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverHandle, stream);

    int info;
    int *d_info = nullptr; /* device error info */

    int Lwork = 0;               /* size of workspace */
    double *Workspace = nullptr; /* device workspace */

    /* step 2: COPY MATRICES TO DEVICE */
    double *Bcuda, *Acuda, *Xcuda;

    cudaMalloc((void **)&Acuda, size * size * sizeof(double));
    cudaMalloc((void **)&Bcuda, size * sizeof(double));
    cudaMalloc((void **)&Xcuda, size * sizeof(double));

    cudaMemcpy(Acuda, Aserialized, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Bcuda, B, size * sizeof(double), cudaMemcpyHostToDevice);

    /* step 3: query working space */
    cusolverDnDpotrf_bufferSize(cusolverHandle, uplo, size, Acuda, size, &Lwork);
    cudaMalloc(&Workspace, sizeof(double) * Lwork);
    cudaMalloc((void **)&d_info, sizeof(int));

    /* step 4: Cholesky factorization */
    cusolverDnDpotrf(cusolverHandle, uplo, size, Acuda, size, Workspace, Lwork, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (0 > info)
    {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    else if (info > 0)
        exit(1);

    /* step 5: solve A*X = b */
    cusolverDnDpotrs(cusolverHandle, uplo, size, 1, Acuda, size, Bcuda, size, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (0 > info)
    {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    cudaMemcpy(X, Bcuda, sizeof(double) * size, cudaMemcpyDeviceToHost);
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

    free(A);
    free(B);
    free(X);
    free(Aserialized);
}

void iterativeRefinement(double **A, double *B, double *X, int size)
{

    double *Aserialized = (double *)malloc(size * size * sizeof(double));
    serializeMatrix(size, A, Aserialized);

    // INITIALIZE CUSOLVER
    cusolverDnHandle_t cusolverHandle;
    cudaStream_t stream = NULL;
    cusolverStatus_t error;

    cusolverDnCreate(&cusolverHandle);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverHandle, stream);

    int info, *dipiv;
    int *d_info = nullptr; /* device error info */
    int *iters = (int *)malloc(sizeof(int));

    size_t Lwork = 0;          /* size of workspace */
    void *Workspace = nullptr; /* device workspace */

    /* step 2: COPY MATRICES TO DEVICE */
    double *Bcuda, *Acuda, *Xcuda;

    cudaMalloc((void **)&Acuda, size * size * sizeof(double));
    cudaMalloc((void **)&Bcuda, size * sizeof(double));
    cudaMalloc((void **)&Xcuda, size * sizeof(double));

    cudaMalloc((void **)&d_info, sizeof(int));
    cudaMalloc((void **)&dipiv, size * size * sizeof(int));

    cudaMemcpy(Acuda, Aserialized, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Bcuda, B, size * sizeof(double), cudaMemcpyHostToDevice);

    /* step 3: query working space */
    error = cusolverDnDHgesv_bufferSize(cusolverHandle, size, 1, Acuda, size, dipiv, Bcuda, size, Xcuda, size, Workspace, &Lwork);
    cudaMalloc(&Workspace, Lwork);
    // printf("Error: %s\n", cudaGetErrorEnum(error));

    /* step 4: Iterative Refinement solution */
    error = cusolverDnDHgesv(cusolverHandle, size, 1, Acuda, size, dipiv, Bcuda, size, Xcuda, size, Workspace, Lwork, iters, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Error: %s\n", cudaGetErrorEnum(error));

    if (0 != info)
    {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    // printf("Size of Workspace: %d\n", Lwork);
    // printf("Iterations: %d\n",*iters);
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

    free(Aserialized);
}


int main()
{
    int size = 273;
    double **A, *B, *Xcalculated, *X;

    // ALLOCATE MEMORY
    A = (double **)malloc(size * sizeof(double *));
    B = (double *)malloc(size * sizeof(double));
    Xcalculated = (double *)malloc(size * sizeof(double));
    X = (double *)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++)
        A[i] = (double *)malloc(size * sizeof(double));

    // READ MATRICES
    readSquareMatrix("A.txt", size, A);
    readVector("B.txt", size, B);
    readVector("X.txt", size, X);

    struct timeval start = tic();

    iterativeRefinement(A, B, Xcalculated, size);

    printf("Iterative refinement time is %f\n", toc(start));

    double thres = 1e-10;
    if (compareVectors(size, X, Xcalculated, thres))
        printf("Solution is True\n");
    else
        printf("Solution is False\n");

    // start = tic();

    // printf("\n\n");

    // Cholesky(A,B,Xcalculated,size);

    // printf("Cholesky factorization time is %f\n",toc(start));

    // if (compareVectors(size, X, Xcalculated,thres))
    //     printf("Solution is True\n");
    // else
    //     printf("Solution is False\n");
}

