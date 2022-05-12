#ifndef READ_H
#define READ_H
#include "types.h"

void readSparseMMMatrix(char *file_path, SparseMatrix *Mtrx);

void saveMatrix(SparseMatrix *res, char *filename);

void saveVector(char *filename, int size, double *array);

void printSparseMatrix(SparseMatrix *res);

void printDenseMatrix(DenseMatrix *mat);

int readSquareMatrix(char *filename, int size, double **array);

int readVector(char *filename, int size, double *array);

void readMMVector(char *filename, Vector *vec);

#endif // READ_H
