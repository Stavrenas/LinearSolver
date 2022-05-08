#include <stdint.h>

#ifndef TYPES_H
#define TYPES_H

typedef struct
{
    double *values;  // element values
    uint32_t *row_idx; // cumulative elements of each row -> last elements equals NNZ
    uint32_t *col_idx; // column indices of the elements 
    uint32_t size;      // Matrix size (assuming square matrices only)
} Matrix;


#endif //TYPES_H
