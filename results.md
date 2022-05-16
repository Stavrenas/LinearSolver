# Iterative refinement results for dense matrices 

The following results are derived from the solution of Ax = B.

For a square 273 * 273 matrix with double precision elements.

For a 1000x1000 matrix we have:

| Type of function   | Main precision | Lowest precision | Workspace Size | Iterations | 100 runs average | min Threshold |
| ------------------ | -------------- | ---------------- | -------------- | ---------- | ---------------- | ------------- |
| cusolverDnIRSXgesv | double         | double           | 8.432.896      | 0          | 548.35 ms        | 1e-14         |
| cusolverDnIRSXgesv | double         | single           | 4.222.976      | 1          | 533.33 ms        | 1e-11         |
| cusolverDnIRSXgesv | double         | half             | 12.609.152     | 1          | 548.72 ms        | 1e-7          |
| CPU                | double         | double           | x              | x          | 15.47 ms         | x             |

For a 4241x4241 matrix we have:

| Type of function   | Main precision | Lowest precision | Workspace Size | Iterations | 100 runs average | min Threshold |
| ------------------ | -------------- | ---------------- | -------------- | ---------- | ---------------- | ------------- |
| cusolverDnIRSXgesv | double         | double           | 145.044.480    | 0          | 1157.1 ms        | 1e-11         |
| cusolverDnIRSXgesv | double         | single           | 72.573.312     | 1          | 613.14 ms        | 1e-6          |
| cusolverDnIRSXgesv | double         | half             | 162.392.960    | 5          | 720.32 ms        | 1e-4          |
| CPU                | double         | double           | x              | x          | 423.22 ms        | x             |

For a 9287x9287 matrix we have:

| Type of function   | Main precision | Lowest precision | Workspace Size | Iterations | 100 runs average |
| ------------------ | -------------- | ---------------- | -------------- | ---------- | ---------------- |
| cusolverDnIRSXgesv | double         | double           | 1.340.254.976  | 0          | 6.21 s           |
| cusolverDnIRSXgesv | double         | single           | 670.164.096    | 5          | 1.13 s           |
| cusolverDnIRSXgesv | double         | half             | 707.650.688    | 50(max)    | 7.54 s           |
| CPU                | double         | double           | x              | x          | 3.51 s           |

## Effect of tolerance

The default tolerance is 1e-8. These results are derived form the last linear system.

### Single precision arithmetic

| Tolerance | Iterations | Total error  | Run time |
| ------------------ | ---------- | ------------ | -------- |
| 1e-8               | 1          | 6.239012e+01 | 1.18 s   |
| 1e-10              | 1          | 6.239012e+01 | 1.18 s   |
| 1e-12              | 2          | 6.239012e+01 | 1.19 s   |
| 1e-14              | 3          | 6.239012e+01 | 1.21 s   |
| 1e-16              | 3          | 6.239012e+01 | 1.21 s   |
| 1e-18              | 50         | 6.239012e+01 | 1.72 s   |

### Half precision arithmetic

In this example, the algorithm did not converge using half precision arithmetic without falling back to double precision

| Tolerance | Iterations | Total error  | Run time |
| ------------------ | ---------- | ------------ | -------- |
| 1e-8               | 50(max)    | 8.130342e+03 | 3.02 s   |
| 1e-10              | 50(max)    | 8.130342e+03 | 3.13 s   |
| 1e-12              | 50(max)    | 8.130342e+03 | 3.36 s   |
| 1e-14              | 50(max)    | 8.130342e+03 | 3.06 s   |
| 1e-16              | 50(max)    | 8.130342e+03 | 3.06 s   |
| 1e-18              | 50(max)    | 8.130342e+03 | 3.17 s   |

If fallback is enabled, the run time is about 9s and the error 6.239012e+01 ("correct").
