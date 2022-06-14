#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "read.h"
#include "mmio.h"
#include "utilities.h"

//**For sparse Matrices**//

void readSparseMMMatrix(char *file_path, SparseMatrix *Mtrx)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *coo_val, *csr_val;

    if ((f = fopen(file_path, "r")) == NULL)
    {
        printf("No such matrix file.\n");
        exit(1);
    }
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(1);

    /* reseve memory for matrices */

    I = (int *)malloc(nz * sizeof(int));
    J = (int *)malloc(nz * sizeof(int));
    coo_val = (double *)malloc(nz * sizeof(double));
    csr_val = (double *)malloc(nz * sizeof(double));

    for (i = 0; i < nz; i++)
    {
        fscanf(f, "%d %d %lf", &I[i], &J[i], &coo_val[i]);
        I[i]--; /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f != stdin)
        fclose(f);

    // Up to this point, I[] cointains row index and J[] column index for the nonzero elements

    const int nnz = nz;

    // printf("M is %d, nnz is %d\n", M, nnz);
    int *row_idx = (int *)malloc((M + 1) * sizeof(int));
    int *col_idx = (int *)malloc(nnz * sizeof(int));

    // Call coo2csr
    coo2csr(row_idx, col_idx, csr_val, I, J, nnz, M, coo_val);

    Mtrx->row_idx = row_idx;
    Mtrx->col_idx = col_idx;
    Mtrx->values = csr_val;
    Mtrx->size = M;
}

void readSystem(char *file_path, SparseMatrix *Mtrx, Vector *B, Vector *X)
{
    FILE *ptr;
    int nrows, nnz;
    double *matrixValues, *vectorValuesB, *vectorValuesX;
    int64_t *row_ptr64;
    int *col_idx, *row_ptr;

    if (strstr(file_path, ".bin"))
    {
        ptr = fopen(file_path, "rb");

        fread(&nrows, sizeof(int), 1, ptr);
        fread(&nnz, sizeof(int), 1, ptr);

        printf("nrows: %d and nnz: %d\n", nrows, nnz);
        Mtrx->size = nrows;
        B->size = nrows;
        X->size = nrows;

        matrixValues = (double *)malloc(nnz * sizeof(double));
        vectorValuesB = (double *)malloc(nrows * sizeof(double));
        vectorValuesX = (double *)malloc(nrows * sizeof(double));
        row_ptr64 = (int64_t *)malloc((nrows + 1) * sizeof(int64_t));
        col_idx = (int *)malloc(nnz * sizeof(int));

        size_t ret = fread(row_ptr64, sizeof(int64_t), nrows + 1, ptr);
        if (ret != nrows + 1)
            printf("Error in read: %ld\n", ret);

        ret = fread(col_idx, sizeof(int), nnz, ptr);
        if (ret != nnz)
            printf("Error in read: %ld\n", ret);

        ret = fread(matrixValues, sizeof(double), nnz, ptr);
        if (ret != nnz)
            printf("Error in read: %ld\n", ret);

        ret = fread(vectorValuesB, sizeof(double), nrows, ptr);
        if (ret != nrows)
            printf("Error in read: %ld\n", ret);

        ret = fread(vectorValuesX, sizeof(double), nrows, ptr);
        if (ret != nrows)
            printf("Error in read: %ld\n", ret);

        row_ptr = (int *)malloc((nrows + 1) * sizeof(int));
        for (int i = 0; i < nrows + 1; i++)
            row_ptr[i] = row_ptr64[i];
    }

    else if (strstr(file_path, ".txt"))
    {
        ptr = fopen(file_path, "r");

        fscanf(ptr, "%d", &nrows);
        fscanf(ptr, "%d", &nnz);

        printf("nrows: %d and nnz: %d\n", nrows, nnz);
        Mtrx->size = nrows;
        B->size = nrows;
        X->size = nrows;

        matrixValues = (double *)malloc(nnz * sizeof(double));
        vectorValuesB = (double *)malloc(nrows * sizeof(double));
        vectorValuesX = (double *)malloc(nrows * sizeof(double));
        row_ptr64 = (int64_t *)malloc((nrows + 1) * sizeof(int64_t));
        col_idx = (int *)malloc(nnz * sizeof(int));

        for (int i = 0; i < nrows + 1; i++)
            fscanf(ptr, "%ld", &row_ptr64[i]);

        for (int i = 0; i < nnz ; i++)
            fscanf(ptr, "%d", &col_idx[i]);

        for (int i = 0; i < nnz; i++)
            fscanf(ptr, "%lf", &matrixValues[i]);

        for (int i = 0; i < nrows; i++)
            fscanf(ptr, "%lf", &vectorValuesB[i]);

        for (int i = 0; i < nrows; i++)
            fscanf(ptr, "%lf", &vectorValuesX[i]);

        row_ptr = (int *)malloc((nrows + 1) * sizeof(int));
        for (int i = 0; i < nrows + 1; i++)
            row_ptr[i] = row_ptr64[i];
    }

    if (ptr == NULL)
    {
        printf("Error finding file\n");
        return;
    }

    Mtrx->values = matrixValues;
    Mtrx->row_idx = row_ptr;
    Mtrx->col_idx = col_idx;

    B->values = vectorValuesB;
    X->values = vectorValuesX;

    fclose(ptr);
}

void saveSystem(char *filename, SparseMatrix *mtr, Vector *B, Vector *X)
{
    FILE *f;
    int n = mtr->size;
    int nnz = mtr->row_idx[mtr->size];
    if ((f = fopen(filename, "w")) == NULL)
        printf("Err\n");
    fprintf(f, "Nrows: %d\n", n);
    fprintf(f, "NNZ: %d\n", nnz);
    fprintf(f, "K row ptr\n");
    for (int i = 0; i <= n; i++)
        fprintf(f, "%d\n", mtr->row_idx[i]);
    fprintf(f, "K col idx\n");
    for (int i = 0; i < nnz; i++)
        fprintf(f, "%d\n", mtr->col_idx[i]);
    fprintf(f, "K values\n");
    for (int i = 0; i < nnz; i++)
        fprintf(f, "%.6f\n", mtr->values[i]);
    fprintf(f, "B\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%.6f\n", B->values[i]);
    fprintf(f, "U\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%.6f\n", X->values[i]);
    fclose(f);
}

void printSparseMatrix(SparseMatrix *res)
{
    int nnz = res->row_idx[res->size];

    if (nnz < 10 || res->size < 10)
    {
        printf("C->col_idx = [ ");
        for (int i = 0; i < nnz; i++)
            printf("%d ", res->col_idx[i]);
        printf("] \n");

        printf("C->row_idx = [ ");
        for (int i = 0; i <= res->size; i++)
            printf("%d ", res->row_idx[i]);
        printf("] \n");

        printf("values = [ ");
        for (int i = 0; i < nnz; i++)
            printf("%e ", res->values[i]);
        printf("] \n");
    }
    else
    {
        int max = 3;
        printf("C->col_idx = [ ");
        for (int i = 0; i < max; i++)
            printf("%d ", res->col_idx[i]);
        printf(" ... ");
        for (int i = nnz - max; i < nnz; i++)
            printf("%d ", res->col_idx[i]);
        printf("] \n");

        printf("C->row_idx = [ ");
        for (int i = 0; i < max; i++)
            printf("%d ", res->row_idx[i]);
        printf(" ... ");
        for (int i = res->size - max + 1; i <= res->size; i++)
            printf("%d ", res->row_idx[i]);
        printf("] \n");

        printf("values = [ ");
        for (int i = 0; i < max; i++)
            printf("%e ", res->values[i]);
        printf(" ... ");
        for (int i = nnz - max; i < nnz; i++)
            printf("%e ", res->values[i]);
        printf("] \n");
    }
}

void printDenseMatrix(DenseMatrix *mat)
{
    int size = mat->size;
    if (size > 10)
        size = 10;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            printf("(%d,%d): %e ", i, j, mat->values[i * mat->size + j]);

        printf("\n");
    }
}

void saveMatrix(SparseMatrix *res, char *filename)
{
    FILE *filepointer = fopen(filename, "w"); // create a binary file
    int nnz = res->row_idx[res->size];

    fprintf(filepointer, "%d %d %d", res->size, res->size, nnz);

    for (int i = 0; i < nnz; i++)
        fprintf(filepointer, "%f ", res->values[i]);
    fprintf(filepointer, "\n");

    for (int i = 0; i < res->size; i++)
        fprintf(filepointer, "%d ", res->col_idx[i]);
    fprintf(filepointer, "\n");

    for (int i = 0; i < res->size; i++)
        fprintf(filepointer, "%d ", res->row_idx[i]);
    fprintf(filepointer, "\n");
    fclose(filepointer);
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
    }

    fclose(pf);

    return 1;
}

//**For vectors**//

int readVector(char *filename, int size, double *array)
{

    FILE *pf;
    pf = fopen(filename, "r");

    if ((pf = fopen(filename, "r")) == NULL)
    {
        printf("No such matrix file.\n");
        exit(1);
    }

    for (size_t i = 0; i < size; i++)
    {
        float temp;
        fscanf(pf, "%e", &temp);
        array[i] = temp;
    }

    fclose(pf);

    return 1;
}

void saveVector(char *filename, int size, double *array)
{
    FILE *f;
    if ((f = fopen(filename, "w")) == NULL)
        printf("Err\n");
    for (int i = 0; i < size; i++)
        fprintf(f, "%e\n", array[i]);
}

void readMMVector(char *filename, Vector *vec)
{

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int size;
    uint32_t i;
    double *val;

    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("No such vector file.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of vector .... */

    if ((ret_code = mm_read_vec_crd_size(f, &size)) != 0)
        exit(1);

    /* reseve memory for values  */

    val = (double *)malloc(size * sizeof(double));

    for (i = 0; i < size; i++)
        fscanf(f, "%lf", &val[i]);

    if (f != stdin)
        fclose(f);

    vec->size = size;
    vec->values = val;
}

void printVector(Vector *v)
{
    int size = v->size;
    int max = 3;
    printf("Values = [ ");
    for (int i = 0; i < max; i++)
        printf("%e ", v->values[i]);
    printf(" ... ");
    for (int i = size - max; i < size; i++)
        printf("%e ", v->values[i]);
    printf("]\n");
}

void printArray(double *values, int size)
{
    Vector *v = (Vector *)malloc(sizeof(Vector));
    v->values = values;
    v->size = size;
    printVector(v);
    free(v);
}
