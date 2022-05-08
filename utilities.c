#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
        if (abs(X_calculated[i] - X[i]) > threshold)
        {
            printf("Error at %d: error is %e\n", i, abs(X_calculated[i] - X[i]));
            return false;
        }
    }

    return true;
}


//**For sparse matrices**//