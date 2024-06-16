#include <mpi.h>

#include "headers/arnoldi_iteration-v1.hpp"
#include "headers/distr_mtx_ops.hpp"

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

int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H) {
    double dotProd = 0;
    double temp = 0;
    V->setCol(0, initVec);

    //auxiliary
    auto * privW = (double *) malloc(counts[me] * sizeof(double));
    auto * w = (double *) malloc(m * sizeof(double));
    double * vCol;

    for(int k = 1; k < k_total + 1; k++) {

        V->getCol(k-1, &vCol);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, privW);

        for(int j = 0; j < k; j++) {
            V->getCol(j, &vCol, displs[me]);

            temp = cblas_ddot(counts[me], privW, 1, vCol, 1);
            MPI_Allreduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            cblas_daxpy(counts[me], -dotProd, vCol, 1, privW, 1);

            H->setValue(j, k-1, dotProd);
        }

        if(k == k_total) break;

        MPI_Allgatherv(privW, helpSize, MPI_DOUBLE, w, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        double wNorm = cblas_dnrm2(m, w, 1.0);

        if(wNorm < EPS52) break;

        H->setValue(k, k - 1, wNorm);
        V->getCol(k, &vCol);
        //fazer isto em cada nó?
        #pragma omp parallel for
        for(int i = 0; i < m; i++){
            vCol[i] = w[i] / wNorm;
        }
    }

    free(w);
    free(privW);

    return k_total;
}