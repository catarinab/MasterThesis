#include <mpi.h>

#include "headers/arnoldi_iteration.hpp"
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
    V->setCol(0, initVec);

    //auxiliary
    auto * privZ = (double *) malloc(counts[me] * sizeof(double));
    auto * z = (double *) malloc(m * sizeof(double));
    memcpy(z, initVec.values.data(), m * sizeof(double));
    double * vCol;

    V->setCol(0, initVec, displs[me], counts[me]);

    for(int k = 0; k < k_total; k++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z, 0.0, privZ);
        for(int j = 0; j <= k; j++) {
            //dense matrix was done to be column major so you only have to get the pointer to initial item
            V->getCol(j, &vCol);

            dotProd = cblas_ddot(counts[me], privZ, 1, vCol, 1);

            MPI_Allreduce(MPI_IN_PLACE, &dotProd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            //w = w - dotProd * vCol
            cblas_daxpy(counts[me], -dotProd, vCol, 1, privZ, 1);

            H->setValue(j, k, dotProd);
        }

        if(k == k_total - 1) break;

        double wNorm = cblas_ddot(counts[me], privZ, 1, privZ, 1);

        MPI_Allreduce(MPI_IN_PLACE, &wNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        wNorm = sqrt(wNorm);

        if(abs(wNorm) < EPS52) break;

        H->setValue(k + 1, k, wNorm);

        V->getCol(k + 1, &vCol);
        for(int i = 0; i < counts[me]; i++){
            vCol[i] = privZ[i] / wNorm;
        }
        MPI_Allgatherv(vCol, counts[me], MPI_DOUBLE, z, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    free(z);
    free(privZ);

    return k_total;
}