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
int arnoldiIteration(csr_matrix A, dense_vector initVec, int k_total, int m, int me, int nprocs, dense_matrix * V,
                     dense_matrix * H) {



    /*int stat = mkl_sparse_set_mv_hint(A.getMKLSparseMatrix(),SPARSE_OPERATION_NON_TRANSPOSE,A.getMKLDescription(),
                                      k_total);

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_set_mv_hint" << endl;
        return 1;
    }

    stat = mkl_sparse_optimize(A.getMKLSparseMatrix());

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_optimize" << endl;
        return 1;
    }*/

    int func = 0;
    int sendEnd = ENDTAG;

    //helper nodes
    while(me != 0) {
        MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if(func == ENDTAG) return 0;
        else if(func > 0)
            helpProcess(A, me, m, func, nprocs, displs, counts);
    }

    V->setCol(0, std::move(initVec));

    int k;

    //auxiliary
    dense_vector opResult(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {
        w = distrMatrixVec(A, V->getCol(k-1), m, me, nprocs);

        for(int j = 0; j < k; j++) {
            dense_vector b = V->getCol(j);
            double dotProd = distrDotProduct(w, b, m, me, nprocs);

            w = distrSumOp(w, b, -dotProd, m, me, nprocs);

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