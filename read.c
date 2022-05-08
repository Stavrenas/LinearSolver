#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "types.h"
#include "read.h"
#include "mmio.h"

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
        temp = fscanf(f, "%d %d", &I[i], &J[i]);
        I[i]--; /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f != stdin)
        fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    // mm_write_banner(stdout, matcode);
    // mm_write_mtx_crd_size(stdout, M, N, nz);
    // Up to this point, I[] cointains row index and J[] column index for the nonzero elements
    /*for (i=0; i<nz; i++)
        {
             fprintf(stdout, "%d %d\n", I[i]+1, J[i]+1);
             printf("%d %d\n", I[i]+1, J[i]+1);
        }
        */

    const uint32_t nnz = nz;

    // printf("M is %d, nnz is %d\n", M, nnz);
    uint32_t *csc_row = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    uint32_t *csc_col = (uint32_t *)malloc((M + 1) * sizeof(uint32_t));
    uint32_t isOneBased = 0;

    // Call coo2csc for isOneBase false
    coo2csc(csc_row, csc_col, I, J, nnz, M, isOneBased);

    Mtrx->csc_elem = csc_col; // csc_col[i] -> total elements up to ith row (size + 1)
    Mtrx->csc_idx = csc_row;  // csc_row[i] -> column index of ith element   (nnz     )

    Mtrx->size = M;
}

void coo2csc(
    uint32_t *const row,           /*!< CSC row start indices */
    uint32_t *const col,           /*!< CSC column indices */
    uint32_t const *const row_coo, /*!< COO row indices */
    uint32_t const *const col_coo, /*!< COO column indices */
    uint32_t const nnz,            /*!< Number of nonzero elements */
    uint32_t const n,              /*!< Number of rows/columns */
    uint32_t const isOneBased      /*!< Whether COO is 0- or 1-based */
)
{
    for (uint32_t l = 0; l < n + 1; l++)
        col[l] = 0;

    for (uint32_t l = 0; l < nnz; l++)
        col[col_coo[l] - isOneBased]++;

    // ----- cumulative sum
    for (uint32_t i = 0, cumsum = 0; i < n; i++)
    {
        uint32_t temp = col[i];
        col[i] = cumsum;
        cumsum += temp;
    }
    col[n] = nnz;
    // ----- copy the row indices to the correct place
    for (uint32_t l = 0; l < nnz; l++)
    {
        uint32_t col_l;
        col_l = col_coo[l] - isOneBased;

        uint32_t dst = col[col_l];
        row[dst] = row_coo[l] + 1;

        col[col_l]++;
    }
    // ----- revert the column pointers
    for (uint32_t i = 0, last = 0; i < n; i++)
    {
        uint32_t temp = col[i];
        col[i] = last;
        last = temp;
    }
}

void printMatrix(Matrix *res)
{
    int nnz = res->row_idx[res->size];
    printf("C->col_idx = [");
    for (int i = 0; i <= nnz; i++)
        printf("%d ", res->col_idx[i]);
    printf("] \n");

    printf("C->row_idx = [");
    for (int i = 0; i < res->size + 1; i++)
        printf("%d ", res->row_idx[i]);
    printf("] \n");

    printf("values = [");
    for (int i = 0; i < res->nnz; i++)
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

    fprintf(filepointer, "%d %d %d",res->size,res->size,nnz);

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

void readMatrix(char *file_path, Matrix *M){

    int M,N,NNZ;



}
