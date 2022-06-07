#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H
#include"types.h"

__global__ void floatToDoubleVector(float *left, double *right, int size);

void gpuLU(SparseMatrix *mat);

void sortSparseMatrix(SparseMatrix *mat);

#endif