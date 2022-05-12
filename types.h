#include <stdint.h>

#ifndef TYPES_H
#define TYPES_H

typedef struct
{
    double *values;  // element values
    int *row_idx; // cumulative elements of each row -> last elements equals NNZ
    int *col_idx; // column indices of the elements 
    int size;      // Matrix size (assuming square matrices only)
} SparseMatrix;

typedef struct
{
    double *values;  // element values
    int size;      // vector size 
} Vector;

typedef struct
{
    double *values;  // element values
    int size;      // matrix size 
} DenseMatrix;


#endif //TYPES_H
