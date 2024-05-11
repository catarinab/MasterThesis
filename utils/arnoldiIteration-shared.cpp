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
    dense_vector w(m);
    double *vCol;
    auto * dotProd = new double[k_total + 1]();
    double wNorm;

    V->getCol(0, &vCol);

    for(k = 1; k < k_total + 1; k++) {
        memset(dotProd, 0, k * sizeof(double));

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, w.values.data());


        #pragma omp parallel shared(dotProd) private(vCol)
        {
            for(int j = 0; j < k; j++) {
                V->getCol(j, &vCol);

                #pragma omp for reduction(+:dotProd[j:j+1])
                for (int i = 0; i < m; i++) {
                    dotProd[j] += (w.values[i] * vCol[i]);
                }

                #pragma omp for simd nowait
                for(int i = 0; i < m; i++) {
                    w.values[i] = w.values[i] - vCol[i] * dotProd[j];
                }
            }
        }
        //H(:, k-1) = dotProd
        for(int i = 0; i < k; i++) {
            H->setValue(i, k - 1, dotProd[i]);
        }

        if(k < k_total) {
            wNorm = cblas_dnrm2(m, w.values.data(), 1);

            H->setValue(k, k - 1, wNorm);

            V->getCol(k, &vCol);
            if(wNorm != 0)
            for (int i = 0; i < m; i++) {
                vCol[i] = w.values[i] / wNorm;
            }
        }
    }

    delete[] dotProd;

    return k;
}