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

void readSystem(char *file_path, SparseMatrix *Mtrx, Vector *B, Vector *X);

void saveSystem(char *filename, SparseMatrix *mtr, Vector *B, Vector *X);

void printVector(Vector *v);

void printArray(double *values, int size);

#endif // READ_H
