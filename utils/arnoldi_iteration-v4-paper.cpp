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
    double temp = 0;
    V->setCol(0, initVec);

    //auxiliary
    auto * privW = (double *) malloc(counts[me] * sizeof(double));
    auto * w = (double *) malloc(m * sizeof(double));
    double * vCol;
    double * recvVCol;
    auto dotProds = new double[k_total]();
    double wDot = 0;
    MPI_Request request;

    V->getCol(0, &vCol);

    for(int k = 1; k < k_total + 1; k++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, privW);

        for (int j = 0; j < k; j++) {
            V->getCol(j, &vCol, displs[me]);

            dotProds[j] = cblas_ddot(counts[me], privW, 1, vCol, 1);
        }

        if (k == k_total) break;

        wDot = cblas_ddot(counts[me], privW, 1, privW, 1);

        MPI_Allreduce(MPI_IN_PLACE, dotProds, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &wDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        fprintf(stderr, "me: %d, wDot: %f\n", me, wDot);

        if(wDot < EPS52) break;

        double hVal = 0;
        double vVal = 0;
        #pragma omp parallel
        {
            #pragma omp for reduction(+:hVal)
            for (int i = 0; i < k; i++) {
                hVal += pow(dotProds[i], 2);
            }
            hVal = wDot - hVal;
            H->setValue(k, k - 1, sqrt(hVal));

            for (int i = 0; i < k; i++) {
                V->getCol(i, &vCol);
                #pragma omp for reduction(+:vVal)
                for(int j = displs[me]; j < displs[me] + counts[me]; j++) {
                    vVal += (privW[j] - vCol[j] * dotProds[j]);
                }
                if(me == 0)
                    fprintf(stderr, "current vVal for k: %d, i: %d: %f\n", k, i, vVal);
            }

            vVal /= hVal;

            if(me == 0) {
                fprintf(stderr, "me: %d, vVal: %f, hVal: %f\n", me, vVal, hVal);
            }
            #pragma omp for
            for (int i = 0; i < counts[me]; i++) {
                privW[i] = (privW[i] - vVal) / hVal;
            }
        }

        V->getCol(k, &vCol);

        MPI_Allgatherv(vCol, counts[me], MPI_DOUBLE, privW, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        fprintf(stderr, "me: %d, done AllGather\n", me);

        //print vCol
        if(me == 0) {
            for (int i = 0; i < m; i++) {
                fprintf(stderr, "vCol[%d]: %f, ", i, vCol[i]);
            }
            fprintf(stderr, "\n");
        }


    }
    free(w);
    free(privW);

    return k_total;
}