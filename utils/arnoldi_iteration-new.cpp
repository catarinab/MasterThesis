#include <mpi.h>

#include "headers/arnoldi_iteration.hpp"
#include "headers/distr_mtx_ops.hpp"
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

int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H) {
    double dotProd = 0;
    V->setCol(0, initVec);

    //auxiliary
    dense_vector privW(counts[me]);
    dense_vector w(m);
    double * vCol;

    for(int k = 1; k < k_total + 1; k++) {

        V->getCol(k-1, &vCol);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, privW.values.data());

        for(int j = 0; j < k; j++) {
            V->getCol(j, &vCol, displs[me]);

            double temp = cblas_ddot(counts[me], privW.values.data(), 1, vCol, 1);
            MPI_Allreduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            cblas_daxpy(counts[me], -dotProd, vCol, 1, privW.values.data(), 1);

            H->setValue(j, k-1, dotProd);
        }

        MPI_Allgatherv(&privW.values[0], helpSize, MPI_DOUBLE, &w.values[0], counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        if(k == k_total) break;

        double wNorm = w.getNorm2();

        if(wNorm < 1e-52)
            return k;

        H->setValue(k, k - 1, wNorm);
        V->getCol(k, &vCol);
        #pragma omp parallel for
        for(int i = 0; i < m; i++){
            vCol[i] = w.values[i] / wNorm;
        }
    }

    return k_total;
}