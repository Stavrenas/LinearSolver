#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utilities.h"

void main()
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
    printf("Cpu time is %f\n",toc(start));

    // printf("info is %d and solution: \n",info);
    //     for(int i = 0; i < size ; i++)
    //         printf("x[%d] = %e\n",i,B[i]);

    if (checkSolution(size, B, X))
        printf("Solution is True\n");
    else
        printf("Solution is False\n");
}