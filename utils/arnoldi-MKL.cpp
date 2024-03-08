#include <utility>

#include "headers/arnoldi-MKL.hpp"
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
int arnoldiIteration(const csr_matrix& A, dense_vector initVec, int k_total, int m, dense_matrix * V, dense_matrix * H) {


    V->setCol(0, std::move(initVec));

    int k = 1;

    //auxiliar
    dense_vector opResult(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {

        w = sparseMatrixVector(A, V->getCol(k-1));
        for(int j = 0; j < k; j++) {
            dense_vector b = V->getCol(j);
            double dotProd = dotProduct(w, b);

            w = addVec(w, b, -dotProd);
            
            H->setValue(j, k-1, dotProd);
        }

        
        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0) 
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    return k;
}