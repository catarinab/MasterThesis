#include <cstring>
#include "headers/arnoldiIteration-shared.hpp"
#include <omp.h>

/*  Parameters
    ----------
    A : An m × m array (csr_matrix)
    b : Initial mx1 vector (dense_vector)
    n : Degree of the Krylov space (int)
    m : Dimension of the matrix (int)

    Returns
    -------
    V : An m x n array (dense_matrix), where the columns are an orthonormal basis of the Krylov subspace.
    H : An n x n array (dense_matrix). A on basis V. It is upper Hessenberg.
*/
int arnoldiIteration(const csr_matrix& A, const dense_vector& initVec, int k_total, int m, dense_matrix * V,
                     dense_matrix * H) {

    int stat = mkl_sparse_set_mv_hint(A.getMKLSparseMatrix(),SPARSE_OPERATION_NON_TRANSPOSE,A.getMKLDescription(),
                                      k_total);

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_set_mv_hint" << endl;
        return 1;
    }

    stat = mkl_sparse_optimize(A.getMKLSparseMatrix());

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_optimize" << endl;
        return 1;
    }

    V->setCol(0, initVec);

    int k;

    //auxiliary
    auto* w = static_cast<double *>(aligned_alloc(64, m * sizeof(double)));
    double *vCol;
    double wNorm;

    for(k = 1; k < k_total + 1; k++) {
        V->getCol(k-1, &vCol);

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, w);

        for(int j = 0; j < k; j++) {
            V->getCol(j, &vCol);

            double dotProd = cblas_ddot(m, w, 1, vCol, 1);

            cblas_daxpy(m, -dotProd, vCol, 1, w, 1);

            H->setValue(j, k - 1, dotProd);
        }

        if(k < k_total){
            double tempNorm = 0;

            #pragma omp parallel shared(w, tempNorm) private(vCol, wNorm)
            {
                #pragma omp for reduction(+:tempNorm)
                for (int i = 0; i < m; i++) {
                    tempNorm += w[i] * w[i];
                }
                wNorm = sqrt(tempNorm);

                if(wNorm != 0) {
                    V->getCol(k, &vCol);
                    //V(:, k) = w / wNorm
                    #pragma omp for
                    for (int i = 0; i < m; i++) {
                        vCol[i] = w[i] / wNorm;
                    }
                }
            }
            H->setValue(k, k - 1, sqrt(tempNorm));
        }


    }

    free(w);

    return k;
}