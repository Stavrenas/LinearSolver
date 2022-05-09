#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "types.h"
#include "read.h"
#include "mmio.h"
#include "utilities.h"

void readMMMatrix(char *file_path, Matrix *Mtrx)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    uint32_t i, *I, *J;
    double *val;

    if ((f = fopen(file_path, "r")) == NULL)
        exit(1);

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
    val = (double *)malloc(nz * sizeof(double));

    int temp; // to supress the warning
    for (i = 0; i < nz; i++)
    {
        temp = fscanf(f, "%d %d %lf", &I[i], &J[i], &val[i]);
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

    // Call coo2csr for isOneBase false
    coo2csr(row_idx, col_idx, I, J, nnz, M);

    Mtrx->row_idx = row_idx;
    Mtrx->col_idx = col_idx;
    Mtrx->values = val;
    Mtrx->size = M;
}

void coo2csr(
    uint32_t *const row_idx,       /*!< CSR row indices */
    uint32_t *const col_idx,       /*!< CSR column indices */
    uint32_t const *const row_coo, /*!< COO row indices */
    uint32_t const *const col_coo, /*!< COO column indices */
    uint32_t const nnz,            /*!< Number of nonzero elements */
    uint32_t const n               /*!< Number of rows/columns */
)
{
    for (uint32_t l = 0; l < nnz; l++)
        col_idx[l] = col_coo[l];

    for (uint32_t i = 0; i <= n; i++)
        row_idx[i] = 0;

    for (uint32_t i = 0; i < n; i++)
        row_idx[row_coo[i]]++;

    // ----- cumulative sum
    for (uint32_t i = 0, cumsum = 0; i < n; i++)
    {
        uint32_t temp = row_idx[i];
        row_idx[i] = cumsum;
        cumsum += temp;
    }
    row_idx[n] = nnz;

    // // ----- copy the row indices to the correct place
    // for (uint32_t l = 0; l < nnz; l++)
    // {
    //     uint32_t col_l;
    //     col_l = col_coo[l];

    //     uint32_t dst = col_idx[col_l];
    //     row_idx[dst] = row_coo[l] + 1;

    //     col_idx[col_l]++;
    // }
    // // ----- revert the column pointers
    // for (uint32_t i = 0, last = 0; i < n; i++)
    // {
    //     uint32_t temp = col_idx[i];
    //     col_idx[i] = last;
    //     last = temp;
    // }
}

void printMatrix(Matrix *res)
{
    int nnz = res->row_idx[res->size];
    printf("C->col_idx = [");
    for (int i = 0; i <= nnz; i++)
        printf("%d ", res->col_idx[i]);
    printf("] \n");

    printf("C->row_idx = [");
    for (int i = 0; i < nnz + 1; i++)
        printf("%d ", res->row_idx[i]);
    printf("] \n");

    printf("values = [");
    for (int i = 0; i < nnz; i++)
        printf("%d ", res->values[i]);
    printf("] \n");
}

void clearMatrix(Matrix *A)
{
    if (A != NULL)
    {
        free(A->values);
        free(A->row_idx);
        free(A->col_idx);
        free(A);
    }
}

void saveMatrix(Matrix *res, char *filename)
{
    FILE *filepointer = fopen(filename, "w"); // create a binary file
    int nnz = res->row_idx[res->size];

    fprintf(filepointer, "%d %d %d", res->size, res->size, nnz);

    for (int i = 0; i < nnz; i++)
        fprintf(filepointer, "%d ", res->values[i]);
    fprintf(filepointer, "\n");

    for (int i = 0; i < res->size; i++)
        fprintf(filepointer, "%d ", res->col_idx[i]);
    fprintf(filepointer, "\n");

    for (int i = 0; i < res->size; i++)
        fprintf(filepointer, "%d ", res->row_idx[i]);
    fprintf(filepointer, "\n");
    fclose(filepointer);
}

void readMatrix(char *file_path, Matrix *Mtrx)
{

    int M, N, NNZ;
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
        values[i] = randomTrueDouble();
    }

    quickSort(cols, 0, nnz - 1);

    for (int i = 0; i < nnz; i++) // sort rows
    {
        int col = cols[i];
        int start = i;
        int end = i;
        for (int j = i; j < nnz; j++)
        {
            if (cols[j] != col)
            {
                end = j - 1;
                break;
            }
        }
        quickSort(rows, start, end);
    }

    for (int i = 0; i < nnz; i++) // delete duplicates
    {
        int col = cols[i];
        int start = i;
        int end = i;
        for (int j = i; j < nnz; j++)
        {
            if (cols[j] != col)
            {
                end = j - 1;
                break;
            }
        }
        if (start + 1 < nnz)
        {
            for (int j = start + 1; j < end; j++)
            {
                if (rows[j] == rows[j - 1])
                {
                    rows[j] = rand() % size + 1;
                    j = start + 1;
                }
            }
        }
        quickSort(rows, start, end);
    }

    for (i = 0; i < nnz; i++)
        fprintf(f, "%d %d %10.3g\n", rows[i], cols[i], values[i]);

    fclose(f);
}
