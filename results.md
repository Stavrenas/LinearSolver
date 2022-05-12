# Results for iterative refinement function

The following results are derived from the solution of Ax = B , where A is a square 273 * 273 matrix with double precision elements.
A needs 596.232 bytes to be stored and B needs 2.184 bytes.

Type of function | Main precision | Lowest precision | Workspace Size | Iterations | Min Threshold | 100 runs time
--- | --- | --- | --- | --- | --- | ---
cusolverDnDDgesv | double | double | 1.140.480 | 1 | 1e-9 | 52.118542
cusolverDnDSgesv | double | single | 806.400 | 2 |1e-9 | 51.058257
cusolverDnDHgesv | double | half | 1.863.552 | 11 |1e-9 | 51.709522
