#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H
#include "types.h"

__global__ void floatToDoubleVector(float *left, double *right, int size);

__global__ void matrixVectorMult(int n, int nnz, double *Avalues, int *rowPtr, int *colIdx, double *Vvalues, double *result);

__global__ void spmv_csr_vector_kernel(const int num_rows, const int *ptr, const int *indices, const double *data, const double *x, double *y);

void gpuLU(SparseMatrix *mat);

void sortSparseMatrix(SparseMatrix *mat);

void residual(SparseMatrix *mat, Vector *B, double *X);

#endif