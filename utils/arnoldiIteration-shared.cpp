#include "headers/arnoldiIteration-shared.hpp"

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
                     dense_matrix * H, int nu) {

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
    dense_vector aux(m);

    for(k = 1; k < k_total + 1; k++) {
        V->getCol(k-1, &aux);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        aux.values.data(), 0.0, w.values.data());

        double dotProd;
        #pragma omp parallel shared(V, H, w, dotProd)
        {
            for(int j = 0; j < k; j++) {
                //dotprod entre w e V->getCol(j)
                #pragma omp for reduction(+:dotProd)
                for (int i = 0; i < m; i++) {
                    dotProd += (w.values[i] * V->getValue(i, j));
                }

                #pragma omp for
                for(int i = 0; i < m; i++) {
                    w.insertValue(i, w.values[i] - V->getValue(i, j) * dotProd);
                }

                #pragma omp single
                {
                    H->setValue(j, k - 1, dotProd);
                    dotProd = 0;
                }
            }
        }


        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0)
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    return k;
}