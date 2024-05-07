#include "headers/arnoldiIteration-shared-nu.hpp"

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
int arnoldiIteration(const csr_matrix &A, const dense_vector &initVec, int k_total, int m, dense_matrix *V,
                     dense_matrix *H, int nu) {

    int stat = mkl_sparse_set_mv_hint(A.getMKLSparseMatrix(),SPARSE_OPERATION_NON_TRANSPOSE,A.getMKLDescription(),
                                      k_total*nu);

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
    dense_vector temp(m);
    double *vCol;

    for(k = 1; k < k_total + 1; k++) {
        V->getCol(k-1, &temp);
        for(int mult = 0; mult < nu; mult++) {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                            temp.values.data(), 0.0, w.values.data());
            if(mult != nu - 1)
                temp.values = w.values;
        }



        double dotProd;
        #pragma omp parallel shared(dotProd, vCol)
        {
            for(int j = 0; j < k; j++) {
                //dotprod entre w e V->getCol(j)
                #pragma omp single
                {
                    dotProd = 0;
                    V->getCol(j, &vCol);
                }

                #pragma omp for simd reduction(+:dotProd)
                for (int i = 0; i < m; i++) {
                    dotProd += (w.values[i] * vCol[i]);
                }

                #pragma omp for simd nowait
                for(int i = 0; i < m; i++) {
                    w.values[i] = w.values[i] - vCol[i] * dotProd;
                }

                #pragma omp single
                    H->setValue(j, k - 1, dotProd);
            }
        }


        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0)
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    return k;
}