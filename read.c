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
    uint32_t i, *I, *J;
    double *coo_val,*csr_val;

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

    I = (uint32_t *)malloc(nz * sizeof(uint32_t));
    J = (uint32_t *)malloc(nz * sizeof(uint32_t));
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

    const uint32_t nnz = nz;

    // printf("M is %d, nnz is %d\n", M, nnz);
    uint32_t *row_idx = (uint32_t *)malloc((M + 1) * sizeof(uint32_t));
    uint32_t *col_idx = (uint32_t *)malloc(nnz * sizeof(uint32_t));

    // Call coo2csr
    coo2csr(row_idx, col_idx,csr_val, I, J, nnz, M,coo_val);

    Mtrx->row_idx = row_idx;
    Mtrx->col_idx = col_idx;
    Mtrx->values = csr_val;
    Mtrx->size = M;
}

void printSparseMatrix(SparseMatrix *res)
{
    int nnz = res->row_idx[res->size];
    printf("C->col_idx = [");
    for (int i = 0; i < nnz; i++)
        printf("%d ", res->col_idx[i]);
    printf("] \n");

    printf("C->row_idx = [");
    for (int i = 0; i <= res->size; i++)
        printf("%d ", res->row_idx[i]);
    printf("] \n");

    printf("values = [");
    for (int i = 0; i < nnz; i++)
        printf("%e ", res->values[i]);
    printf("] \n");
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
