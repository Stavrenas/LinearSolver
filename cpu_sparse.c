#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utilities.h"
#include "read.h"

int main1(int argc, char **argv)
{

    Matrix *A = (Matrix *)malloc(sizeof(Matrix));
    Matrix *B = (Matrix *)malloc(sizeof(Matrix));
    Matrix *X = (Matrix *)malloc(sizeof(Matrix));

    char *filenameA = (char *)malloc(40 * sizeof(char));
    char *filenameB = (char *)malloc(40 * sizeof(char));
    char *filenameX = (char *)malloc(40 * sizeof(char));

    if (argc == 3)
    {
        sprintf(filenameA, "%s.mtx", argv[1]);
        sprintf(filenameB, "%s.mtx", argv[2]);
    }
    else
    {
        printf("Usage: ./cpu_sparse A_Matrix B_Vector X_vector(optional) \n");
        exit(-1);
    }

    readMMMatrix(filenameA, A);
    readMMMatrix(filenameB, B);

    // serializeMatrix(size, A, Aserialized);

    int info;
    char uplo = 'U';
    // int sizeB = size;
    int sides = 1;
    struct timeval start = tic();
    // LAPACK_dpotrf(&uplo, &size, Aserialized, &size, &info);
    // LAPACK_dpotrs(&uplo, &size, &sides, Aserialized, &size, B, &sizeB, &info);
    printf("Cpu time is %f\n", toc(start));

    // printf("info is %d and solution: \n",info);
    //     for(int i = 0; i < size ; i++)
    //         printf("x[%d] = %e\n",i,B[i]);

    if (argc == 3)
    {
        // print solution?
    }
    else
    {
        Matrix *Xcorrect = (Matrix *)malloc(sizeof(Matrix));
        char *filenameXcorrect = (char *)malloc(40 * sizeof(char));
        sprintf(filenameXcorrect, "%s.mtx", argv[3]);
        // if (checkSolution(Xcorrect, X) == 1)
        //     printf("Solution is True\n");
        // else
        //     printf("Solution is False\n");
    }
}

int main()
{

    char *matrixName = "Test"; 
    char *filename = (char *)malloc(40 * sizeof(char));
    char *filenameB = (char *)malloc(40 * sizeof(char));
    char *filenameX = (char *)malloc(40 * sizeof(char));

    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

    sprintf(filename, "%s.mtx", matrixName);
    sprintf(filenameB, "%s-B.txt", matrixName);
    sprintf(filenameX, "%s-X.txt", matrixName);

    //generateMMMatrix(filename, 100, 1000);
    readMMMatrix(filename, mat);
    //printMatrix(mat);

    double* B = (double*)malloc(mat->size*sizeof(double));
    double* X = (double*)malloc(mat->size*sizeof(double));

    generateSolutionVector(matrixName, mat);
    readVector(filenameB, mat->size, B);
    readVector(filenameX, mat->size, X);
    
}