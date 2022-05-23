# Iterative refinement results for sparse matrices

Nvidia white [paper](https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html)
## Solution Pipeline

- Allocate GPU memory
- Copy data to GPU memory
- Setup matrix descriptors
- Incomplete LU Factorization (**float**)
  1. Calculate buffer size
  2. Analysis
  3. "Solve"
- Copy result and typecast to double on gpu
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

## Results

For a 1000x1000 matrix we have:

| Tolerance | Iterations | Run time |
| --------- | ---------- | -------- |
| 1e-4      | 127        | 68 ms    |
| 1e-5      | 205        | 88 ms    |
| 1e-6      | 299        | 113 ms   |
| 1e-7      | 397        | 137 ms   |
| 1e-8      | 494        | 162 ms   |
| 1e-9     | 592        | 188 ms   |
| 1e-10     | 689        | 213 ms   |
| 1e-11     | 787        | 244 ms   |
| 1e-12     | 884        | 267 ms   |
| 1e-13     | 982        | 326 ms   |
