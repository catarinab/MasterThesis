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

    for(k = 1; k < k_total + 1; k++) {
        double tempNorm = 0;
        V->getCol(k-1, &vCol);
        memset(dotProd, 0, k * sizeof(double));

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, w.values.data());


        #pragma omp parallel shared(dotProd) private(vCol, wNorm)
        {
            for(int j = 0; j < k; j++) {
                V->getCol(j, &vCol);

                #pragma omp for reduction(+:dotProd[j:j+1])
                for (int i = 0; i < m; i++) {
                    dotProd[j] += (w.values[i] * vCol[i]);
                }

                #pragma omp for nowait
                for(int i = 0; i < m; i++) {
                    w.values[i] = w.values[i] - vCol[i] * dotProd[j];
                }
            }

            if(k < k_total) {
                #pragma omp for reduction(+:tempNorm)
                for (int i = 0; i < m; i++) {
                    tempNorm += w.values[i] * w.values[i];
                }

                wNorm = sqrt(tempNorm);
                if(wNorm != 0) {
                    V->getCol(k, &vCol);
                    #pragma omp for
                    for (int i = 0; i < m; i++) {
                        vCol[i] = w.values[i] / wNorm;
                    }
                }
            }

            //H(:, k-1) = dotProd
        }
        if(k < k_total)
            H->setValue(k, k - 1, sqrt(tempNorm));

        for (int i = 0; i < k; i++) {
            H->setValue(i, k - 1, dotProd[i]);
        }


    }

    delete[] dotProd;

    return k;
}