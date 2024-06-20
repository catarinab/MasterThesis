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
 * Section 4, Algorithm 4
 * KSPPGMRES
 */
int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H) {

    //auxiliary
    auto * w = (double *) malloc(counts[me] * sizeof(double));
    dense_matrix Z(counts[me], k_total + 1);
    auto * z_i = (double *) malloc(m * sizeof(double));
    memcpy(z_i, initVec.values.data(), m * sizeof(double));
    double vDot = 0, prevH;
    double * vCol, *prevVCol, * zCol;
    auto updateVec = new double[counts[me]]();
    auto dotProds = new double[k_total + 2]();
    MPI_Request requestZ, requestH, requestHDot;

    V->setCol(0, initVec, displs[me], counts[me]);
    Z.setCol(0, initVec, displs[me], counts[me]);

    for(int i = 0; i <= k_total; i++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z_i, 0.0, w);

        if(i > 0) {
            MPI_Wait(&requestH, MPI_STATUS_IGNORE);
            for (int idx = 0; idx <= i - 1; idx++) {
                H->setValue(idx, i - 1, dotProds[idx]);
            }
        }

        if(i > 1) {
            prevH = sqrt(dotProds[i]);
            if(abs(prevH) < EPS52) break;
            H->setValue(i - 1, i - 2, prevH);

            V->getCol(i - 1, &vCol);
            Z.getCol(i, &zCol);

            #pragma omp parallel for
            for(int idx = 0; idx < counts[me]; idx++) {
                vCol[idx] = vCol[idx] / prevH;
                zCol[idx] = z_i[idx + displs[me]] / prevH;
                w[idx] = w[idx] / prevH;
            }

            for (int j = 0; j <= i - 2; j++) {
                H->setValue(j, i - 1, H->getValue(j, i - 1) / prevH);
            }
            H->setValue(i - 1, i - 1, H->getValue(i - 1, i - 1) / pow(prevH, 2));
        }

        if(i == k_total) break;

        memset(updateVec, 0, counts[me] * sizeof(double));
        #pragma omp parallel for reduction(+:updateVec[:counts[me]]) private(vCol)
        for (int j = 0; j <= i - 1; j++) {
            double hVal = H->getValue(j, i - 1);
            Z.getCol(j + 1, &zCol);
            for (int idx = 0; idx < counts[me]; idx++) {
                updateVec[idx] += hVal * zCol[idx];
            }
        }

        Z.getCol(i + 1, &zCol);
        #pragma omp parallel for
        for (int idx = 0; idx < counts[me]; idx++) {
            zCol[idx] = w[idx] - updateVec[idx];
        }

        MPI_Iallgatherv(zCol, counts[me], MPI_DOUBLE, z_i, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &requestZ);

        if (i > 0) {
            memset(updateVec, 0, counts[me] * sizeof(double));
            #pragma omp parallel for reduction(+:updateVec[:counts[me]]) private(vCol)
            for (int j = 0; j <= i - 1; j++) {
                double hVal = H->getValue(j, i - 1);
                V->getCol(j, &vCol);
                for (int idx = 0; idx < counts[me]; idx++) {
                    updateVec[idx] += hVal * vCol[idx];
                }
            }

            V->getCol(i, &vCol);
            Z.getCol(i, &zCol);

            #pragma omp parallel for
            for (int idx = 0; idx < counts[me]; idx++) {
                vCol[idx] = zCol[idx] - updateVec[idx];
            }

            dotProds[i + 1] = cblas_ddot(counts[me], vCol, 1, vCol, 1);
        }

        Z.getCol(i + 1, &zCol);
        for(int j = 0; j <= i; j++) {
            V->getCol(j, &prevVCol);
            dotProds[j] = cblas_ddot(counts[me], zCol, 1, prevVCol, 1);
        }
        MPI_Iallreduce(MPI_IN_PLACE, dotProds, i + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &requestH);
        MPI_Wait(&requestZ, MPI_STATUS_IGNORE);
    }
    free(z_i);
    free(w);

    delete[] updateVec;
    delete[] dotProds;

    return k_total;
}