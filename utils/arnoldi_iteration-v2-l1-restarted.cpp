#include <mpi.h>

#include "headers/arnoldi_iteration.hpp"
#include "headers/mtx_ops_mkl.hpp"
#include "headers/calculate_MLF.hpp"
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
 * Paper: Hiding Global Communication Latency in the GMRES Algorithm on Massively Parallel Machines
 * Section 3, Algorithm 2
 * */

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix& V, const dense_matrix& mlfH, double betaVal) {
    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

double restartedArnoldiIteration_MLF(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                              dense_matrix * H, double alpha, double beta) {
    int k = 0;
    double approx = 0;

    while(true) {
        H->resize(k_total - k);
        V->resizeCols(k_total - k);
        int currK = arnoldiIteration(A, initVec, k_total - k, m, me, V, H);
        H->resize(currK);
        V->resizeCols(currK);
        dense_matrix mlfH(currK, currK);
        if(me == 0) {
            cout << "k: " << k << endl;
            cout << "currK: " << currK << endl;
            mlfH = calculate_MLF((double *) H->getDataPointer(), alpha, beta, currK);
        }
        k += currK;

        MPI_Bcast(mlfH.getValues(), currK * currK, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        dense_vector res = getApproximation(*V, mlfH, 1);

        double temp = cblas_ddot(counts[me], res.values.data(), 1, res.values.data(), 1);

        MPI_Allreduce(MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        approx += sqrt(temp);
        if (k >= k_total - 1) break;
        V->getLastCol(initVec);
    }

    return approx;
}


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

    int k = 0;

    for(k = 0; k < k_total; k++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z, 0.0, privZ);

        for (int j = 0; j <= k; j++) {
            V->getCol(j, &vCol);
            dotProds[j] = cblas_ddot(counts[me], privZ, 1, vCol, 1);
        }

        if (k == k_total - 1) break;

        double temp  = cblas_ddot(counts[me], privZ, 1, privZ, 1);
        MPI_Allreduce(&temp, &wDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(sqrt(wDot) < EPS52) break;

        MPI_Allreduce(MPI_IN_PLACE, dotProds, k + 1 , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double hVal = 0;
        for (int i = 0; i <= k; i++) {
            hVal += pow(dotProds[i], 2);
        }

        //Check for breakdown and restart or reorthogonalize if necessary
        if(wDot - hVal <= 0) {
            cout << "Breakdown at k = " << k << endl;
            break;
        }

        hVal = sqrt(wDot - hVal);

        memset(vValVec, 0, counts[me] * sizeof(double));
        #pragma omp parallel private(vCol)
        {
            #pragma omp for reduction(+:vValVec[:counts[me]]) private(vCol)
            for (int i = 0; i <= k; i++) {
                V->getCol(i, &vCol);
                for(int j = 0; j < counts[me]; j++) {
                    vValVec[j] += vCol[j] * dotProds[i];
                }
            }
            V->getCol(k + 1, &vCol);
            #pragma omp for
            for (int i = 0; i < counts[me]; i++) {
                vCol[i] = (privZ[i] - vValVec[i]) / hVal;
            }
        }

        V->getCol(k + 1, &vCol);
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

    return k;
}