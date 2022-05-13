#include <cblas.h>
#include "lapacke.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utilities.h"
#include "read.h"

void smallExample()
{
    int size = 273;
    double **A, *B, *X, *Aserialized;

    A = (double **)malloc(size * sizeof(double *));
    B = (double *)malloc(size * sizeof(double));
    X = (double *)malloc(size * sizeof(double));
    Aserialized = (double *)malloc(size * size * sizeof(double));
    for (int i = 0; i < size; i++)
        A[i] = (double *)malloc(size * sizeof(double));

    readSquareMatrix("A.txt", size, A);
    readVector("B.txt", size, B);
    readVector("X.txt", size, X);

    serializeMatrix(size, A, Aserialized);

    int info;
    char uplo = 'U';
    int sizeB = size;
    int sides = 1;
    struct timeval start = tic();
    LAPACK_dpotrf(&uplo, &size, Aserialized, &size, &info);
    LAPACK_dpotrs(&uplo, &size, &sides, Aserialized, &size, B, &sizeB, &info);
    printf("Cpu time is %f\n", toc(start));

    // printf("info is %d and solution: \n",info);
    //     for(int i = 0; i < size ; i++)
    //         printf("x[%d] = %e\n",i,B[i]);

    if (compareVectors(size, B, X, 1e-10))
        printf("Solution is True\n");
    else
        printf("Solution is False\n");
}

int main()
{
    char *matrixName = "data/e30r0100";
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

    double *Bcopy = (double *)malloc(B->size * sizeof(double));

    for (int i = 0; i < B->size; i++)
        Bcopy[i] = B->values[i];

    int info, size, sides, *ipiv;
    ipiv = (int *)malloc(size * sizeof(int));
    size = dense->size;
    sides = 1;
    struct timeval start = tic();

    LAPACK_dgesv(&size, &sides, dense->values, &size, ipiv, Bcopy, &size, &info); // X vector is saved on Bcopy

    if (info != 0)
        printf("Info is %d\n", info);

    printf("Cpu time is %f\n", toc(start));

    sparseToDense(sparse, dense);        // overwrite factorized matrix to get original values for evaluation
    checkSolutionDense(dense, B, Bcopy); // calculate |Ax-b|

    // clearDense(dense);
    // clearSparse(sparse);
    // clearVector(B);
    // free(Bcopy);

    return 0;
}