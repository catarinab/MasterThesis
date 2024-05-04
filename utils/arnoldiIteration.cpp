#include <mpi.h>

#include <utility>

#include "headers/arnoldiIteration.hpp"
#include "headers/distr_mtx_ops.hpp"
#include "headers/help_process.hpp"

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
int arnoldiIteration(const csr_matrix& A, const dense_vector& initVec, int k_total, int m, int me, int nprocs, dense_matrix * V,
                     dense_matrix * H, int nu) {

    int func = 0;
    int sendEnd = ENDTAG;

    //helper nodes
    while(me != 0) {
        MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if(func == ENDTAG) return 0;
        else if(func > 0)
            helpProcess(A, me, m, func, displs, counts);
    }

    V->setCol(0, initVec);

    int k;

    //auxiliary
    dense_vector b(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {

        V->getCol(k-1, &w);
        for(int mult = 0; mult < nu; mult ++)
            w = distrMatrixVec(A, w, m);

        for(int j = 0; j < k; j++) {
            V->getCol(j, &b);
            double dotProd = distrDotProduct(w, b, m, me);

            w = distrSumOp(w, b, -dotProd, m, me);

            H->setValue(j, k-1, dotProd);            
        }

        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0) 
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    MPI_Bcast(&sendEnd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return k;
}