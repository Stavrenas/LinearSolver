#ifndef UTILITIES_H
#define UTILITIES_H
#include <sys/time.h>
#include <stdbool.h>
#include "types.h"

struct timeval tic();

double toc(struct timeval begin);

void serializeMatrix(int size, double **A, double *serialized);

int findSum(int size);

bool compareVectors(int size, double Bcalculated[], double B[], double threshold);

void swap(int *a, int *b);

int partition(int arr[], int low, int high);

void quickSort(int arr[], int low, int high);

double randomTrueDouble();

void generateSolutionVector(char *matrixName, SparseMatrix *Mtr);

void createMatrix(SparseMatrix *res, char *filename);

void coo2csr(
    int *const row_idx, /*!< CSR row indices */
    int *const col_idx, /*!< CSR column indices */
    double *const csr_val,
    int const *const row_coo, /*!< COO row indices */
    int const *const col_coo, /*!< COO column indices */
    int const nnz,            /*!< Number of nonzero elements */
    int const n,              /*!< Number of rows/columns */
    double *const coo_val);

void clearMatrix(SparseMatrix *A);

void generateMMMatrix(char *filepath, int size, int nnz);

void checkSolutionSparse(SparseMatrix *mtrx, Vector *B, double *X, double thres);

void checkSolutionDense(DenseMatrix *mtrx, Vector *B, double *X, double thres);

void sparseToDense(SparseMatrix *spr, DenseMatrix *dns);

void clearDense(DenseMatrix *matrix);

void clearSparse(SparseMatrix *matrix);

void clearVector(Vector *vec);

int maxInt(int a, int b);

#endif
