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
     - $beta = {p_i \over p_{i-1}}$
     - $p = z + beta \* p$
  5. Compute **$q = Ap$** (**double**)
  6. a = $p_i \over p^Tq$
  7. $x += ap$
  8. $r -= aq$
- Transfer solution to host memory

## Sparse GC Results

For a 31.287 x 31.287 matrix with 2.467.643 nnz _(n10k)_ we have:

Sort and ILU time is 1.35 sec.

Norm is residual norm divided by right-hand side norm

#### GC single vs double

| Norm  | Iterations (Double) | Run time(Double) | Iterations(Float) | Run time(Float) |
| ----- | ------------------- | ---------------- | ----------------- | --------------- |
| 1e-7  | 137                 | 3.13s            | 126               | 2.46 s          |
| 1e-8  | 148                 | 3.13 s           | 140               | 2.51 s          |
| 1e-9  | 160                 | 3.13 s           | 150               | 2.54 s          |
| 1e-10 | 174                 | 3.28 s           | 165               | 2.58s           |
| 1e-11 | 182                 | 2.98 s           | 191               | 2.65 s          |
| 1e-12 | 191                 | 3.5 s            | 202               | 2.69 s          |
| 1e-13 | 200                 | 3.61 s           | 212               | 2.71 s          |


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

nrows: 31.287 and nnz: 2.467.643
| Mode | Time | Residual Norm |
|---|---|---|
| CPU (mkl) | 1sec | 1e-15 |
|GPU direct | 35 sec | 0 |
|GPU iterative simple - Double | 50sec | 1e-12 |
|GPU iterative simple - Single | >120sec | - |
|GPU iterative GC Double | 3 sec | 1e-12 |
|GPU iterative GC Single | 2.68 sec | 1e-12 |

**na_bc.bin**

nrows: 323.856 and nnz: 13.180.992
| Mode | Time | Residual Norm |
|---|---|---|
| CPU(mkl) | 10sec | 1e-8 |
|GPU direct | >120sec | - |
|GPU iterative simple Double | >120sec | - |
|GPU iterative simple Single | >120sec | - |
|GPU iterative GC Double | >120sec | - |
|GPU iterative GC Single | >120sec | - |

${|b| \over |r|} = 5 \* 1e-8$


**Note:** All gpu implementations cannot achieve convergence.

**thermal.txt**

nrows: 82.704 and nnz: 1.173.296
| Mode | Time | Residual Norm |
| --- | --- | --- |
| CPU(mkl) | 1.7 sec | 1e-7 |
|GPU direct | 61.9 sec | 0 |
|GPU iterative simple Double | >120 sec | - |
|GPU iterative simple Single | >120 sec | - |
|GPU iterative GC Double| 2.22 sec | 1e-12 |
|GPU iterative GC Single| >120sec | - |
