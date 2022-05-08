#ifndef READ_H
#define READ_H
#include "types.h"

void readMMMatrix(char *file_path, Matrix *Mtrx);

void readMatrix(char *file_path, Matrix *Mtx);

void saveMatrix(Matrix *res, char *filename);

void printMatrix(Matrix *res);

void createMatrix(Matrix *res, char *filename);

void coo2csc(
    uint32_t *const row,           /*!< CSC row start indices */
    uint32_t *const col,           /*!< CSC column indices */
    uint32_t const *const row_coo, /*!< COO row indices */
    uint32_t const *const col_coo, /*!< COO column indices */
    uint32_t const nnz,            /*!< Number of nonzero elements */
    uint32_t const n,              /*!< Number of rows/columns */
    uint32_t const isOneBased      /*!< Whether COO is 0- or 1-based */
);

void clearMatrix(Matrix *A);

void generateMMMatrix(char* filepath, int size, int nnz);


#endif //READ_H
