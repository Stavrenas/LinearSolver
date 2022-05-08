#ifndef UTILITIES_H
#define UTILITIES_H
#include <sys/time.h>
#include <stdbool.h>

struct timeval tic();

double toc(struct timeval begin);

int readSquareMatrix(char *filename, int size, double **array);

int readVector(char *filename, int size, double *array);

void serializeMatrix(int size, double **A, double *serialized);

int findSum(int size);

bool checkSolution(int size, double X_calculated[], double X[]);

void swap(int* a, int* b);

int partition (int arr[], int low, int high);

void quickSort(int arr[], int low, int high);

double randomTrueDouble();

#endif
