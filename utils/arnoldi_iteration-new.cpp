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

    V->setCol(0, initVec);

    int k;

    //auxiliary
    dense_vector vectorVCol(m);
    dense_vector tempW(counts[me]);
    dense_vector w(m);
    double * vCol;

    double dotProd = 0;

    for(k = 1; k < k_total + 1; k++) {

        V->getCol(k-1, &vectorVCol);

        //cada um tem a sua parte de w
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vectorVCol.values.data(), 0.0, tempW.values.data());

        for(int j = 0; j < k; j++) {
            V->getCol(j, &vCol, displs[me]);

            double temp = cblas_ddot(counts[me], tempW.values.data(), 1, vCol, 1);
            MPI_Allreduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            cblas_daxpy(counts[me], -dotProd, vCol, 1, tempW.values.data(), 1);

            H->setValue(j, k-1, dotProd);
        }

        MPI_Allgatherv(&tempW.values[0], helpSize, MPI_DOUBLE, &w.values[0], counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        if(k == k_total) return k;

        double wNorm = w.getNorm2();

        if(wNorm == 0)
            return k;

        H->setValue(k, k - 1, wNorm);
        V->getCol(k, &vCol);
        #pragma omp parallel for
        for(int i = 0; i < m; i++){
            vCol[i] = w.values[i] / wNorm;

        }
    }
    return k;
}