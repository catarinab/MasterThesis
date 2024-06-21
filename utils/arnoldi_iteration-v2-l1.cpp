#include <mpi.h>

#include "headers/arnoldi_iteration.hpp"
#include "headers/distr_mtx_ops.hpp"

/*   Parameters
 *   ----------
 *   A : An m × m array (csr_matrix)
 *   b : Initial mx1 vector (dense_vector)
 *   n : Degree of the Krylov space (int)
 *   m : Dimension of the matrix (int)
 *
 *   Returns
 *   -------
 *   V : An m x n array (dense_matrix), where the columns are an orthonormal basis of the Krylov subspace.
 *   H : An n x n array (dense_matrix). A on basis V. It is upper Hessenberg.
 */


/*
 * Paper: Hiding Global Communication Latency in the GMRES Algorithm on Massively Parallel Machines
 * Section 3, Algorithm 2
 * */
int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H) {

    //auxiliary
    auto * privZ = (double *) malloc(counts[me] * sizeof(double));
    auto * z = (double *) malloc(m * sizeof(double));
    memcpy(z, initVec.values.data(), m * sizeof(double));
    double * vCol;
    auto vValVec = new double[counts[me]]();
    auto dotProds = new double[k_total]();
    double wDot = 0;
    MPI_Request r1;

    V->setCol(0, initVec, displs[me], counts[me]);

    // k_new = k_old - 1
    // k_old = k_new + 1

    for(int k = 0; k <= k_total; k++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z, 0.0, privZ);

        for (int j = 0; j <= k; j++) {
            V->getCol(j, &vCol);
            dotProds[j] = cblas_ddot(counts[me], privZ, 1, vCol, 1);
        }

        if (k == k_total) break;

        double temp  = cblas_ddot(counts[me], privZ, 1, privZ, 1);
        MPI_Allreduce(&temp, &wDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(abs(sqrt(wDot))< EPS52) break;

        MPI_Allreduce(MPI_IN_PLACE, dotProds, k + 1 , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double hVal = 0;
        for (int i = 0; i <= k; i++) {
            hVal += pow(dotProds[i], 2);
        }

        //Check for breakdown and restart or reorthogonalize if necessary
        if(wDot - hVal <= 0) {
            fprintf(stderr, "Should be Restarting Arnoldi iteration at k = %d\n", k);
            return k;
        }

        hVal = sqrt(wDot - hVal);

        memset(vValVec, 0, counts[me] * sizeof(double));
        #pragma omp parallel for reduction(+:vValVec[:counts[me]]) private(vCol)
        for (int i = 0; i <= k; i++) {
            V->getCol(i, &vCol);
            for(int j = 0; j < counts[me]; j++) {
                vValVec[j] += vCol[j] * dotProds[i];
            }
        }
        V->getCol(k + 1, &vCol);
        for (int i = 0; i < counts[me]; i++) {
            vCol[i] = (privZ[i] - vValVec[i]) / hVal;
        }

        MPI_Iallgatherv(vCol, counts[me], MPI_DOUBLE, z, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &r1);

        if(me == 0) {
            for (int i = 0; i <= k; i++) {
                H->setValue(i, k, dotProds[i]);
            }
            H->setValue(k + 1, k, hVal);
        }

        MPI_Wait(&r1, MPI_STATUS_IGNORE);

    }

    MPI_Allreduce(MPI_IN_PLACE, dotProds, k_total - 1 , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(me == 0) {
        for (int i = 0; i < k_total; i++) {
            H->setValue(i, k_total - 1, dotProds[i]);
        }
    }
    free(z);
    free(privZ);

    delete[] vValVec;
    delete[] dotProds;

    return k_total;
}