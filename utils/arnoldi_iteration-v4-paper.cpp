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

/*
 * Paper: HIDING GLOBAL COMMUNICATION LATENCY IN THE GMRES ALGORITHM ON MASSIVELY PARALLEL MACHINES
 * Section 3, Algorithm 2
 * */
int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H) {
    //auxiliary
    auto * privZ = (double *) malloc(counts[me] * sizeof(double));
    auto * z = (double *) malloc(m * sizeof(double));
    double * vCol;
    auto vValVec = new double[m]();
    auto dotProds = new double[k_total]();
    double wDot = 0;
    MPI_Request request;

    V->setCol(0, initVec);
    V->getCol(0, &vCol);

    for(int k = 1; k < k_total + 1; k++) {

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, privZ);

        for (int j = 0; j < k; j++) {
            V->getCol(j, &vCol, displs[me]);
            dotProds[j] = cblas_ddot(counts[me], vCol, 1, privZ, 1);
        }

        if (k == k_total) break;

        wDot = cblas_ddot(counts[me], privZ, 1, privZ, 1);
        MPI_Allreduce(MPI_IN_PLACE, &wDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, dotProds, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Iallgatherv(privZ, counts[me], MPI_DOUBLE, z, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &request);

        double hVal = 0;
        for (int i = 0; i < k; i++) {
            hVal += pow(dotProds[i], 2);
        }

        //Check for breakdown and restart or reorthogonalize if necessary
        if(wDot - hVal <= 0) {
            fprintf(stderr, "Should be Restarting Arnoldi iteration at k = %d\n", k);
            return k;
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            wDot = sqrt(wDot);
            #pragma omp parallel for
            for(int i = 0; i < m; i++) {
                z[i] /= wDot;
                V->setValue(i, 0, z[i]);
            }
            V->getCol(0, &vCol);
            k = 0;
            continue;
        }
        hVal = sqrt(wDot - hVal);
        H->setValue(k, k - 1, hVal);

        memset(vValVec, 0, k_total * sizeof(double));
        double vVal = 0;
        #pragma omp parallel for reduction(+:vValVec[:m]) private(vCol)
        for (int i = 0; i < k; i++) {
            V->getCol(i, &vCol);
            for(int j = 0; j < m; j++) {
                vValVec[j] += vCol[j] * dotProds[i];
            }
        }

        for (int i = 0; i < k; i++) {
            H->setValue(i, k-1, dotProds[i]);
        }

        V->getCol(k, &vCol);

        cout << "vVal: " << vVal << ", hVal: " << hVal << endl;

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        for (int i = 0; i < m; i++) {
            vCol[i] = (z[i] - vValVec[i]) / hVal;
        }

        cout << "vCol in iteration " << k << ": ";


    }
    free(z);
    free(privZ);

    delete[] vValVec;
    delete[] dotProds;

    return k_total;
}