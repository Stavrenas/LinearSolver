# Iterative refinement results for sparse matrices

Nvidia white [paper](https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html)

## Solution Pipeline

- Allocate GPU memory
- Copy data to GPU memory
- Setup matrix descriptors
- Sort column indices on GPU (for intel MKL dcsrilu02)
- Incomplete LU Factorization (**double** - using intel MKL)
- Setup matrix descriptors for the triangular solvers
- Loop:
  1. Solve **Mz = r** (**double**)
  2. Calculate $p_i = r^T z$ (**double**)
  3. If (i == 0)
     - p = z
  4. else
     - $beta = {p*i \over p*{i-1}}$
     - $p = z + beta \* p$
  5. Compute **q = Ap** (**double**)
  6. a = $p_i \over p^Tq$
  7. $x += ap$
  8. $r -= aq$
- Transfer solution to host memory

## Sparse GC Results

For a 31.287 x 31.287 matrix with 2.467.643 nnz _(n10k)_ we have:

Sort and ILU time is 1.35 sec.

Norm is residual norm divided by right-hand side norm

#### Naive approach vs GC

| Norm  | Iterations | Run time | Iterations(GC) | Run time(GC) |
| ----- | ---------- | -------- | -------------- | ------------ |
| 1e-7  | 4806       | 26.18 s  | 137            | 3.13s        |
| 1e-8  | 6006       | 32.37 s  | 148            | 3.13 s       |
| 1e-9  | 7206       | 38.42 s  | 160            | 3.13 s       |
| 1e-10 | 8406       | 44.48 s  | 174            | 3.28 s       |
| 1e-11 | 9606       | 50.59 s  | 182            | 2.98 s       |
| 1e-12 | 10806      | 56.33 s  | 191            | 3.5 s        |
| 1e-13 | 12006      | 62.43 s  | 200            | 3.61 s       |

For na_bc sort and ILU time is 2.15sec (323.856x323.856 and 13.180.992 nnz)

_Sorting the matrix before **reduces** the total iterations_

### Splitting of total time

The following metrics are derived from the naive approach.

Data type: `Double`
| Step | Percentage |
| ------------------ | ---------- |
| Matrix-vector mult | 8.5% |
| Triangular Solver | 86.2% |

Data type: `Single`
| Step | Percentage |
| ------------------ | ---------- |
| Matrix-vector mult | 6.2% |
| Triangular Solver | 88.3% |

_With single precision the algorithm did **not** converge_

### CPU vs GPU Times

**n10k.bin**
| mode | time | Residual Norm |
|---|---|---|
| CPU (mkl) | 1sec | 1e-15 |
|GPU iterative simple | 50sec | 1e-12 |
|GPU direct | 35 sec | 0 |
|GPU iterative GC | 3 sec | 1e-12 |

**na_bc.bin**
| mode | time | Residual Norm |
|---|---|---|
| CPU(mkl) | 10sec | 1e-8 |
|GPU iterative simple | - | - |
|GPU direct | - | - |
|GPU iterative GC | - | - |

**Note:** All gpu implementations cannot achieve congergence.

Maybe because "The preconditioner matrix M has to be symmetric positive-definite and fixed" ?

<!-- ![alt text](/results/iters.png)

![alt text](/results/run%20time.png) -->
