#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include "utilities.h"

//**For timing**//

struct timeval tic()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv;
}

double toc(struct timeval begin)
{
    struct timeval end;
    gettimeofday(&end, NULL);
    double stime = ((double)(end.tv_sec - begin.tv_sec) * 1000) +
                   ((double)(end.tv_usec - begin.tv_usec) / 1000);
    stime = stime / 1000;
    return (stime);
}

//**For dense matrices**//

int readSquareMatrix(char *filename, int size, double **array)
{

    FILE *pf;
    pf = fopen(filename, "r");

    if (pf == NULL)
        return 0;

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            float temp;
            fscanf(pf, "%e", &temp);
            array[i][j] = temp;
        }
        char *lol;
        fscanf(pf, "\n", lol);
    }

    fclose(pf);

    return 1;
}

int readVector(char *filename, int size, double *array)
{

    FILE *pf;
    pf = fopen(filename, "r");

    if (pf == NULL)
        return 0;

    for (size_t i = 0; i < size; i++)
    {
        float temp;
        fscanf(pf, "%e", &temp);
        array[i] = temp;
    }

    fclose(pf);

    return 1;
}

void serializeMatrix(int size, double **A, double *serialized)
{

    int elements = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            serialized[elements] = A[i][j];
            elements++;
        }
    }
}

int findSum(int size)
{
    int result = 0;

    for (int i = 1; i <= size; i++)
        result += i;
    return result;
}

bool checkSolution(int size, double X_calculated[], double X[])
{

    double threshold = 1e-12;

    for (int i = 0; i < size; i++)
    {
        if (fabs(X_calculated[i] - X[i]) > threshold)
        {
            printf("Error at %d: error is %e\n", i, fabs(X_calculated[i] - X[i]));
            return false;
        }
    }

    return true;
}

bool checkSolutionThres(int size, double X_calculated[], double X[], double threshold)
{

    for (int i = 0; i < size; i++)
    {
        if (fabs(X_calculated[i] - X[i]) > threshold)
        {
            printf("Error at %d: error is %e\n X[%d] = %e and Xcalculated[%d] = %e \n", i, fabs(X_calculated[i] - X[i]), i,X[i],i,X_calculated[i]);
            return false;
        }
    }

    return true;
}

// A utility function to swap two elements
void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partition(int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    int i = (low - 1);     // Index of smaller element and indicates the right position of pivot found so far

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

double randomTrueDouble()
{
    double f;
    int sign, exp, i;
    unsigned int mant;

    char s[33];
    for (i = 0; i < 32; i++)
    {
        if (i == 1)
            continue;
        s[i] = rand() % 2 + '0';
    }

    s[1] = '0';
    s[32] = 0;

    sign = s[0] - '0';

    exp = 0;
    for (i = 1; i <= 8; i++)
        exp = exp * 2 + (s[i] - '0');

    exp -= 127;

    if (exp > -127)
    {
        mant = 1; // The implicit "1."
        exp -= 23;
    }
    else
    {
        mant = 0;
        exp = -126;
        exp -= 23;
    }

    for (i = 9; i <= 31; i++)
        mant = mant * 2 + (s[i] - '0');

    f = mant;

    while (exp > 0)
        f *= 2, exp--;

    while (exp < 0)
        f /= 2, exp++;

    if (sign)
        f = -f;

    return f;
}

//**For sparse matrices**//

void generateSolutionVector(char *matrixName, Matrix *Mtr) // generates B where Ax = B and X is a random vector
{

    FILE *fileB, *fileX;

    double *X = (double *)malloc(Mtr->size * sizeof(double));
    double *B = (double *)malloc(Mtr->size * sizeof(double));
    double temp = 1e-20;
    for (int i = 0; i < Mtr->size; i++)
    {
        do
        {
            temp = randomTrueDouble();
        } while (abs(temp) < 1e-5 || abs(temp) > 1e15);

        X[i] = temp;
        B[i] = 0;
    }

    for (int i = 0; i < Mtr->size; i++)
    {
        int start = Mtr->row_idx[i];
        int end = Mtr->row_idx[i+1];

        for (int j = start; j < end; j++)
            B[i] += X[Mtr->col_idx[j]] * Mtr->values[j];
    }

    char *filenameB = (char *)malloc(40 * sizeof(char));
    char *filenameX = (char *)malloc(40 * sizeof(char));

    sprintf(filenameB, "%s-B.txt", matrixName);
    sprintf(filenameX, "%s-X.txt", matrixName);

    if ((fileB = fopen(filenameB, "w")) == NULL)
        printf("Err\n");

    if ((fileX = fopen(filenameX, "w")) == NULL)
        printf("Err\n");

    for (int i = 0; i < Mtr->size; i++)
        fprintf(fileB, "%e\n", B[i]);
    fclose(fileB);

    for (int i = 0; i < Mtr->size; i++)
        fprintf(fileX, "%e\n", X[i]);
    fclose(fileX);

    free(X);
    free(B);
}

void saveVector(char *filename, int size, double *array)
{
    FILE *f;
    if ((f = fopen(filename, "w")) == NULL)
        printf("Err\n");
    for (int i = 0; i < size; i++)
        fprintf(f, "%e\n", array[i]);
}