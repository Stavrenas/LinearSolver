#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H
#include <cusolverDn.h>
#include <cusparse_v2.h>

__global__ void floatToDoubleVector(float *left, double *right);

#endif