#include "headers/arnoldiIteration.hpp"
#include "headers/mtx_ops.hpp"

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
int arnoldiIteration(csr_matrix A, dense_vector b, int k_total, int m, dense_matrix * V, dense_matrix * H) {


    V->setCol(0, b);

    int k = 1;

    //auxiliar
    dense_vector opResult(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {
        w = sparseMatrixVector(A, V->getCol(k-1), 0, m, m);

        for(int j = 0; j < k; j++) {
            H->setValue(j, k-1, dotProduct(w, V->getCol(j), 0, m));
            opResult = V->getCol(j) * H->getValue(j, k-1);
            w = subtractVec(w, opResult, 0, m);
        }

        
        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0) 
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    return k;
}