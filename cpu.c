
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utilities.h"
#include "read.h"

#include "mkl.h"
#include "mkl_dss.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

// Define the format to printf MKL_INT values
#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif

// void solveSystem(DenseMatrix *dense, Vector *B, double *X)
// {
//     double *Bcopy = (double *)malloc(B->size * sizeof(double));

//     for (int i = 0; i < B->size; i++)
//         Bcopy[i] = B->values[i];

//     int info, size, sides, *ipiv;
//     size = dense->size;
//     ipiv = (int *)malloc(size * sizeof(int));
//     sides = 1;

//     LAPACK_dgesv(&size, &sides, dense->values, &size, ipiv, Bcopy, &size, &info); // X vector is saved on Bcopy

//     if (info != 0)
//         printf("Info is %d\n", info);

//     for (int i = 0; i < size; i++)
//         X[i] = Bcopy[i];

//     free(Bcopy);
// }

void solveSystemSparse(SparseMatrix *mat, Vector *B, double *X, double tolerance)
{
    // convert csr to csc
    int nnz = mat->row_idx[mat->size];

    MKL_INT job[6];

    for (int i = 0; i < 6; i++)
        job[i] = 0;

    job[0] = 0; // convert csr to csc
    job[1] = 0; // csr is 0-based
    job[2] = 1; // csc is 1-based
    job[5] = 1; // fill all arrays for the output storage

    MKL_INT m = mat->size;
    MKL_INT info;
    MKL_INT *ja = (MKL_INT *)mkl_malloc(nnz * sizeof(MKL_INT), 64);
    MKL_INT *ia = (MKL_INT *)mkl_malloc((m + 1) * sizeof(MKL_INT), 64);

    for (int i = 0; i < nnz; i++)
        ja[i] = mat->col_idx[i];

    for (int i = 0; i < m + 1; i++)
        ia[i] = mat->row_idx[i];

    double *acsc = (double *)malloc(nnz * sizeof(double));
    MKL_INT *ja1 = (MKL_INT *)mkl_calloc(nnz, sizeof(MKL_INT), 64);
    MKL_INT *ia1 = (MKL_INT *)mkl_calloc((m + 1), sizeof(MKL_INT), 64);

    mkl_dcsrcsc(job, &m, mat->values, ja, ia, acsc, ja1, ia1, &info);

    mkl_free(ia);
    mkl_free(ja);

    /* Matrix data in CSC format. */

    MKL_INT mtype = 11; /* Real unsymmetric matrix */
    // Descriptor of main sparse matrix properties
    struct matrix_descr descrA;
    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t csrA;
    /* RHS and solution vectors. */
    double bs[m], res, res0;
    MKL_INT nrhs = 1; /* Number of right hand sides. */
    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    void *pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    MKL_INT i, j;
    double ddum;  /* Double dummy */
    MKL_INT idum; /* Integer dummy. */
                  /* -------------------------------------------------------------------- */
                  /* .. Setup Pardiso control parameters. */
                  /* -------------------------------------------------------------------- */
    for (i = 0; i < 64; i++)
        iparm[i] = 0;

    iparm[0] = 1;   /* No solver default */
    iparm[1] = 2;   /* Fill-in reordering from METIS */
    iparm[3] = 0;   /* No iterative-direct algorithm */
    iparm[4] = 0;   /* No user fill-in reducing permutation */
    iparm[5] = 0;   /* Write solution into x */
    iparm[6] = 0;   /* Not in use */
    iparm[7] = 2;   /* Max numbers of iterative refinement steps */
    iparm[8] = 0;   /* Not in use */
    iparm[9] = 13;  /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;  /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;  /* Conjugate/transpose solve */
    iparm[12] = 1;  /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13] = 0;  /* Output: Number of perturbed pivots */
    iparm[14] = 0;  /* Not in use */
    iparm[15] = 0;  /* Not in use */
    iparm[16] = 0;  /* Not in use */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[19] = 0;  /* Output: Numbers of CG Iterations */
    maxfct = 1;     /* Maximum number of numerical factorizations. */
    mnum = 1;       /* Which factorization to use. */
    msglvl = 0;     /* Print statistical information  */
    error = 0;      /* Initialize error flag */
                    /* -------------------------------------------------------------------- */
                    /* .. Initialize the internal solver memory pointer. This is only */
                    /* necessary for the FIRST call of the PARDISO solver. */
                    /* -------------------------------------------------------------------- */
    for (i = 0; i < 64; i++)
        pt[i] = 0;

    /* -------------------------------------------------------------------- */
    /* .. Reordering and Symbolic Factorization. This step also allocates */
    /* all memory that is necessary for the factorization. */
    /* -------------------------------------------------------------------- */
    phase = 11;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &m, acsc, ia1, ja1, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: " IFORMAT, error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors = " IFORMAT, iparm[17]);
    printf("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);
    /* -------------------------------------------------------------------- */
    /* .. Numerical factorization. */
    /* -------------------------------------------------------------------- */
    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &m, acsc, ia1, ja1, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during numerical factorization: " IFORMAT, error);
        exit(2);
    }
    printf("\nFactorization completed ... ");

    /* -------------------------------------------------------------------- */
    /* .. Solution phase. */
    /* -------------------------------------------------------------------- */
    phase = 33;

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, m, m, ia1, ia1 + 1, ja1, acsc);

    // Transpose solve is used for systems in CSC format
    iparm[11] = 2;

    printf("\n\nSolving the system in CSC format...\n");
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &m, acsc, ia1, ja1, &idum, &nrhs, iparm, &msglvl, B->values, X, &error);
    if (error != 0)
    {
        printf("\nERROR during solution: " IFORMAT, error);
        exit(3);
    }

    printf("\nThe solution of the system is: ");
    for (j = 0; j < 5; j++)
    {
        printf("\n x [" IFORMAT "] = % f", j, X[j]);
    }
    printf("\n");
    // Compute residual
    // the CSC format for A is the CSR format for A transposed
    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csrA, descrA, X, 0.0, bs);
    res = 0.0;
    res0 = 0.0;
    for (j = 0; j < m; j++)
    {
        res += (bs[j] - B->values[j]) * (bs[j] - B->values[j]);
        res0 += B->values[j] * B->values[j];
    }
    res = sqrt(res) / sqrt(res0);
    printf("\nRelative residual = %e", res);
    // Check residual
    if (res > 1e-10)
    {
        printf("Error: residual is too high!\n");
    }
    mkl_sparse_destroy(csrA);

    /* -------------------------------------------------------------------- */
    /* .. Termination and release of memory. */
    /* -------------------------------------------------------------------- */
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &m, &ddum, ia1, ja1, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

    mkl_free(ia1);
    mkl_free(ja1);
    free(acsc);
}

int main(int argc, char *argv[])
{
    char *matrixName;

    if (argc == 2)
        matrixName = argv[1];
    else
        matrixName = "data/na_bc.bin";
    SparseMatrix *sparse = (SparseMatrix *)malloc(sizeof(SparseMatrix));
    Vector *B = (Vector *)malloc(sizeof(Vector));
    Vector *Xcorrect = (Vector *)malloc(sizeof(Vector));

    readSystem(matrixName, sparse, B, Xcorrect);

    double *X = (double *)malloc(B->size * sizeof(double));
    // printSparseMatrix(sparse);

    struct timeval start = tic();
    solveSystemSparse(sparse, B, X, 100);

    printf("\nCpu time is %f\n", toc(start));

    saveVector("var/CpuX.txt", B->size, X);
    return 0;
}