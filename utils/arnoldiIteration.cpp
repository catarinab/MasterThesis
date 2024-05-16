#include <mpi.h>

#include "headers/arnoldiIteration.hpp"
#include "headers/distr_mtx_ops.hpp"
#include "headers/help_process.hpp"
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
int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, int nprocs, dense_matrix * V,
                     dense_matrix * H) {

    int func = 0;
    int sendEnd = ENDTAG;

    //helper nodes
    if(me != 0)
        return helpProcess(A, me, m, func, displs, counts);

    V->setCol(0, initVec);

    int k;

    //auxiliary
    dense_vector b(m);
    dense_vector w(m);
    double * vCol;

    for(k = 1; k < k_total + 1; k++) {
        V->getCol(k-1, &b);
        distrMatrixVec(A, b, w, m);

        for(int j = 0; j < k; j++) {
            V->getCol(j, &b);
            double dotProd = distrDotProduct(w, b, m, me);
            distrSumOp(w, b, -dotProd, m, me);
            H->setValue(j, k-1, dotProd);
        }

        if(k == k_total) break;

        double wNorm = vec2norm(w);
        H->setValue(k, k - 1, wNorm);

        if(wNorm != 0) {
            V->getCol(k, &vCol);
            #pragma omp parallel for
            for(int i = 0; i < m; i++){
                vCol[i] = w.values[i] / wNorm;
            }
        }
    }
    MPI_Bcast(&sendEnd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return k;
}