#include "headers/arnoldiIteration-shared.hpp"
#include "headers/mtx_ops_mkl.hpp"

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
    dense_vector opResult(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {

        // w = A^nu *V->getCol(k-1)
        w = V->getCol(k-1);
        for(int mult = 0; mult < nu; mult ++) {
            w = sparseMatrixVector(A, w);
        }


        double dotProd;
        dense_vector b;
        #pragma omp parallel shared(V, H, w, b, dotProd)
        {
        for(int j = 0; j < k; j++) {

            #pragma omp single
            {
                dotProd = 0;
                b = V->getCol(j);
            }

            //dotprod entre w e V->getCol(j)
            #pragma omp for simd reduction(+:dotProd)
            for (int i = 0; i < m; i++) {
                dotProd += (w.values[i] * b.values[i]);
            }

            #pragma omp for simd nowait
            for(int i = 0; i < m; i++) {
                double newVal = b.values[i] * dotProd;
                w.insertValue(i, w.values[i] - newVal);
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