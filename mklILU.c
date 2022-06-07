/* C source code is found in dgemm_example.c */

#define min(x, y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "types.h"

int test()
{
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    m = 2000, k = 200, n = 1000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n",
           m, k, k, n);
    alpha = 1.0;
    beta = 0.0;

    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
           " performance \n\n");
    A = (double *)mkl_malloc(m * k * sizeof(double), 64);
    B = (double *)mkl_malloc(k * n * sizeof(double), 64);
    C = (double *)mkl_malloc(m * n * sizeof(double), 64);
    if (A == NULL || B == NULL || C == NULL)
    {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf(" Intializing matrix data \n\n");
    for (i = 0; i < (m * k); i++)
    {
        A[i] = (double)(i + 1);
    }

    for (i = 0; i < (k * n); i++)
    {
        B[i] = (double)(-i - 1);
    }

    for (i = 0; i < (m * n); i++)
    {
        C[i] = 0.0;
    }

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    printf("\n Computations completed.\n\n");

    printf(" Top left corner of matrix A: \n");
    for (i = 0; i < min(m, 6); i++)
    {
        for (j = 0; j < min(k, 6); j++)
        {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (i = 0; i < min(k, 6); i++)
    {
        for (j = 0; j < min(n, 6); j++)
        {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (i = 0; i < min(m, 6); i++)
    {
        for (j = 0; j < min(n, 6); j++)
        {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf(" Example completed. \n\n");
    return 0;
}

void mklIncompleteLU(SparseMatrix *mat)
{
    const MKL_INT n = mat->size;
    MKL_INT ierr = 0;
    const MKL_INT *ipar = (MKL_INT *)mkl_malloc(128 * sizeof(MKL_INT), 32);
    const double *dpar = (double *)mkl_malloc(128 * sizeof(double), 32);
    int nnz = mat->row_idx[mat->size];

    MKL_INT *ia = (MKL_INT *)mkl_malloc((n + 1) * sizeof(MKL_INT), 64);
    MKL_INT *ja = (MKL_INT *)mkl_malloc((nnz) * sizeof(MKL_INT), 64);

    double *values = (double *)malloc(nnz * sizeof(double));

    // 0-based to 1-based

    for (int i = 0; i <= n; i++)
        ia[i] = mat->row_idx[i] + 1;
    for (int i = 0; i < nnz; i++)
        ja[i] = mat->col_idx[i] + 1;

    dcsrilu0(&n, mat->values, ia, ja, values, ipar, dpar, &ierr);
    printf("error is %lld\n", ierr);

    //overwritte values
    for (int i = 0; i < nnz; i++)
        mat->values[i] = values[i];

    // 1-based to 0-based

    for (int i = 0; i <= n; i++)
        mat->row_idx[i] = ia[i] - 1;
    for (int i = 0; i < nnz; i++)
        mat->col_idx[i] = ja[i] - 1;

    free(values);
}
