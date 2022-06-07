# Iterative refinement results for sparse matrices

Nvidia white [paper](https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html)

## Solution Pipeline

- Allocate GPU memory
- Copy data to GPU memory
- Setup matrix descriptors
- Sort column indices on GPU (for intel MKL dcsrilu02)
- Incomplete LU Factorization (**double** - using intel MKL)
- Setup matrix descriptors for the triangular solvers
- Solve initial system **Ax = b** by solving **Ly = b** and then **Ux = y**
- Loop:
  1. Calculate residual r = b - Ax (**double**)
  2. Find |b| , |r| (**double**)
  3. Break if |r| / |b| < _tolerance_ (**double**)
  4. Solve **Ac = r** by solving **Ly = b** and then **Uc = y** (**double**)
  5. Update solution vector: Xn+1 = Xn + c (**double**)
  6. Restore overwritten vectors
- Transfer solution to host memory

## Sparse Results

For a 1000x1000 matrix we have:

| Tolerance | Iterations | Run time |
| --------- | ---------- | -------- |
| 1e-4      | 127        | 68 ms    |
| 1e-5      | 205        | 88 ms    |
| 1e-6      | 299        | 113 ms   |
| 1e-7      | 397        | 137 ms   |
| 1e-8      | 494        | 162 ms   |
| 1e-9      | 592        | 188 ms   |
| 1e-10     | 689        | 213 ms   |
| 1e-11     | 787        | 244 ms   |
| 1e-12     | 884        | 267 ms   |
| 1e-13     | 982        | 326 ms   |

For a 31287x31287 matrix with 2.467.643 nnz n10k we have:

| Tolerance | Iterations | Run time |
| --------- | ---------- | -------- |
| 1e-4      | 1300       | 8.2 s    |
| 1e-5      | 2413       | 14.15 s  |
| 1e-6      | 3608       | 20.17 s  |
| 1e-7      | 4806       | 26.18 s  |
| 1e-8      | 6006       | 32.37 s  |
| 1e-9      | 7206       | 38.42 s  |
| 1e-10     | 8406       | 44.48 s  |
| 1e-11     | 9606       | 50.59 s  |
| 1e-12     | 10806      | 56.33 s  |
| 1e-13     | 12006      | 62.43 s  |

Sort and ILU time is 1.35 sec.

For na_bc sort and ILU time is 2.15 (323856x323856 and 13.180.992 nnz)

*Sorting the matrix before reduces the total iterations*


![alt text](/results/iters.png)

![alt text](/results/run%20time.png)
