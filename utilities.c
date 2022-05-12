#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "mmio.h"
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

bool compareVectors(int size, double left[], double right[], double threshold)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(left[i] - right[i]) > threshold)
        {
            printf("Error at %d: error is %e ", i, fabs(left[i] - right[i]));
            printf("Left %e, right %e\n", left[i], right[i]);
            //return false;
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

void generateSolutionVector(char *matrixName, SparseMatrix *Mtr) // generates B where Ax = B and X is a random vector
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
        int end = Mtr->row_idx[i + 1];

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

void generateMMMatrix(char *filepath, int size, int nnz)
{
    if (nnz > size * size)
        return;

    FILE *f;
    MM_typecode matcode;
    int *rows = (int *)malloc(nnz * sizeof(int));
    int *cols = (int *)malloc(nnz * sizeof(int));
    double *values = (double *)malloc(nnz * sizeof(double));
    int i;

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    if ((f = fopen(filepath, "w")) == NULL)
        printf("Err\n");

    mm_write_banner(f, matcode);
    mm_write_mtx_crd_size(f, size, size, nnz);

    srand(time(NULL));

    for (i = 0; i < nnz; i++)
    {

        rows[i] = rand() % size + 1;
        cols[i] = rand() % size + 1;
        double temp = 1e-11;
        do
        {
            temp = randomTrueDouble();
        } while (fabs(temp) > 1 || fabs(temp) < 1e-15);
        values[i] = temp * 1e10;
    }

    quickSort(cols, 0, nnz - 1); // sort cols

    for (int i = 0; i < nnz; i++) // sort rows
    {
        int col = cols[i];
        int start = i;
        int end = i;
        for (int j = i; j < nnz; j++)
        {
            if (j == nnz - 1)
            {
                end = j;
                break;
            }
            if (cols[j] != col)
            {
                end = j - 1;
                break;
            }
        }
        quickSort(rows, start, end);
        i = end;
    }

    // for (int i = 0; i < nnz; i++) // delete duplicates
    // {
    //     int col = cols[i];
    //     int start = i;
    //     int end = i;
    //     for (int j = i; j < nnz; j++)
    //     {
    //         if (cols[j] != col)
    //         {
    //             end = j - 1;
    //             break;
    //         }
    //     }

    //     quickSort(rows, start, end);

    //     for (int j = start + 1; j < end; j++)
    //     {
    //         if (rows[j] == rows[j - 1])
    //         {
    //             //printf("dup: rows[%d] = rows [%d] = %d, cols[%d] = cols [%d] = %d\n", j, j - 1, rows[j], j, j - 1, cols[j]);
    //             rows[j] = rand() % size + 1;
    //             j = start + 1;
    //         }
    //     }

    //     quickSort(rows, start, end);
    // }

    // TODO: make matrix diagonally dominant

    for (i = 0; i < nnz; i++)
        fprintf(f, "%d %d %10.3g\n", rows[i], cols[i], values[i]);

    fclose(f);
}

void clearMatrix(SparseMatrix *A)
{
    if (A != NULL)
    {
        free(A->values);
        free(A->row_idx);
        free(A->col_idx);
        free(A);
    }
}

void coo2csr(
    uint32_t *const row_idx, /*!< CSR row indices */
    uint32_t *const col_idx, /*!< CSR column indices */
    double *const csr_val,
    uint32_t const *const row_coo, /*!< COO row indices */
    uint32_t const *const col_coo, /*!< COO column indices */
    uint32_t const nnz,            /*!< Number of nonzero elements */
    uint32_t const n,              /*!< Number of rows/columns */
    double *const coo_val)
{
    for (uint32_t l = 0; l < nnz; l++)
        col_idx[l] = col_coo[l];

    for (uint32_t i = 0; i <= n; i++)
        row_idx[i] = 0;

    for (uint32_t i = 0; i < nnz; i++)
    {
        row_idx[row_coo[i]]++;
        // printf("row_coo[%d] = %d, row_idx[%d] = %d \n",i,row_coo[i],i,row_idx[i]);
    }

    // ----- cumulative sum
    // printf("cumsum\n");
    for (uint32_t i = 0, cumsum = 0; i < n; i++)
    {
        uint32_t temp = row_idx[i];
        row_idx[i] = cumsum;
        // printf("row_idx[%d] = %d \n",i,row_idx[i]);
        cumsum += temp;
    }
    row_idx[n] = nnz;

    // copy column indexes to correct position
    for (int i = 0; i < nnz; i++)
    {
        int row = row_coo[i];
        int dest = row_idx[row];

        col_idx[dest] = col_coo[i];
        csr_val[dest] = coo_val[i];

        row_idx[row]++;
    }

    // revert row pointer
    for (int i = 0, last = 0; i <= n; i++)
    {
        int temp = row_idx[i];
        row_idx[i] = last;
        last = temp;
    }
}

void checkSolutionSparse(SparseMatrix *mtrx, Vector *B, double *X)
{

    double *Bcalculated = (double *)malloc(B->size * sizeof(double));

    /* Calculate Ax = B , given solution vector X*/
    for (int i = 0; i < mtrx->size; i++)
    {
        int start = mtrx->row_idx[i];
        int end = mtrx->row_idx[i + 1];
        Bcalculated[i] = 0;

        for (int j = start; j < end; j++)
            Bcalculated[i] += X[mtrx->col_idx[j]] * mtrx->values[j];
    }

    /* Compare Bcalculated with true B */
    compareVectors(B->size, Bcalculated, B->values, 1e-5);
}

void checkSolutionDense(DenseMatrix *mtrx, Vector *B, double *X)
{
    int size = mtrx->size;
    double *Bcalculated = (double *)malloc(size * sizeof(double));

    /* Calculate Ax = B , given solution vector X*/
    for (int i = 0; i < size; i++)
    {
        Bcalculated[i] = 0;
        //printf("B[%d] = ",i);
        for (int j = 0; j < size; j++){
            Bcalculated[i] += mtrx->values[i * size + j] * X[j];
            //printf(" %.2f * %.2f +",mtrx->values[i * size + j], X[j]);
            //if(mtrx->values[i*size+j]!=0)
                //printf("X is %f ",mtrx->values[i*size+j]);
            
        }
        //printf("\n");
    }

    /* Compare Bcalculated with true B */
    compareVectors(B->size, Bcalculated, B->values, 1e-5);
}

void sparseToDense(SparseMatrix *spr, DenseMatrix *dns)
{
    int size = spr->size;
    double *values = (double *)malloc(size * size * sizeof(double));

    /*Initialize values vector*/
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
            values[row * size + col] = 0;
    }

    /*Populate vector*/
    for (int row = 0; row < spr->size; row++)
    {
        int start = spr->row_idx[row];
        int end = spr->row_idx[row + 1];

        for (int j = start; j < end; j++)
        {
            int col = spr->col_idx[j];
            values[row * size + col] = spr->values[j];
            // printf("[%d][%d]   ", row, col);
        }
    }

    dns->values = values;
    dns->size = size;
}