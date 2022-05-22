void help()
{
    // Suppose that A is m x m sparse matrix represented by CSR format,
    // Assumption:
    // - handle is already created by cusparseCreate(),
    // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
    // - d_x is right hand side vector on device memory,
    // - d_y is solution vector on device memory.
    // - d_z is intermediate result on device memory.

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_M = 0;
    csrsv2Info_t info_L = 0;
    csrsv2Info_t info_U = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha = 1.;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has unit diagonal
    // - matrix U is base-1
    // - matrix U is upper triangular
    // - matrix U has non-unit diagonal
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateCsrilu02Info(&info_M);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseDcsrilu02_bufferSize(handle, m, nnz,
                                 descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz,
                               descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz,
                               descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);

    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on U
    // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    cusparseDcsrilu02_analysis(handle, m, nnz, descr_M,
                               d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
                               policy_M, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
                             d_csrVal, d_csrRowPtr, d_csrColInd,
                             info_L, policy_L, pBuffer);

    cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
                             d_csrVal, d_csrRowPtr, d_csrColInd,
                             info_U, policy_U, pBuffer);

    // step 5: M = L * U
    cusparseDcsrilu02(handle, m, nnz, descr_M,
                      d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
                          d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
                          d_x, d_z, policy_L, pBuffer);

    // step 7: solve U*y = z
    cusparseDcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U,
                          d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
                          d_z, d_y, policy_U, pBuffer);

    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyCsrilu02Info(info_M);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_U);
    cusparseDestroy(handle);
}