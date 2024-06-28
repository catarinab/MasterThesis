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
    l: latency hiding parameter (int)

    Returns
    -------
    V : An m x n array (dense_matrix), where the columns are an orthonormal basis of the Krylov subspace.
    H : An n x n array (dense_matrix). A on basis V. It is upper Hessenberg.
*/

/*
 * Paper: Hiding Global Communication Latency in the GMRES Algorithm on Massively Parallel Machines
 * Section 4, Algorithm 3
 * */

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix& V, const dense_matrix& mlfH, double betaVal) {
    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

dense_vector restartedArnoldiIteration_MLF(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                                           dense_matrix * H, double alpha, double beta, double betaNormB, int l) {
    int k = 0;

    dense_vector b(m);
    dense_vector temp(counts[me]);


    while(true) {
        H->resize(k_total - k);
        V->resizeCols(k_total - k);
        int currK = arnoldiIteration(A, initVec, k_total - k, m, me, V, H, l);
        int size = (currK - k_total) >= 0 ? k_total : currK - l;
        H->resize(size);
        V->resizeCols(size);
        dense_matrix mlfH(size, size);

        if(me == 0) {
            mlfH = calculate_MLF((double *) H->getDataPointer(), alpha, beta, size);
        }

        MPI_Bcast(mlfH.getValues(), size * size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        dense_vector res = getApproximation(*V, mlfH, betaNormB);

        cblas_daxpy(counts[me], 1, res.values.data(), 1, temp.values.data(), 1);

        if (k >= k_total) break;
        V->getLastCol(initVec);
    }

    MPI_Gatherv(&temp.values[0], counts[me], MPI_DOUBLE, &b.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    return b;
}


int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H, int l) {
    //auxiliary
    auto * privZ = (double *) malloc(counts[me] * sizeof(double));
    auto * z = (double *) malloc(m * sizeof(double));
    memcpy(z, initVec.values.data(), m * sizeof(double));
    dense_matrix Z(counts[me], k_total + l + 2);
    dense_matrix G(counts[me], k_total + l + 2);
    double * vCol, * zCol, *xZcol;
    auto vValVec = new double[counts[me]]();
    auto dotProds = new double[k_total + l + 2];
    MPI_Request r1, r2;

    V->setCol(0, initVec, displs[me], counts[me]);
    Z.setCol(0, initVec, displs[me], counts[me]);
    G.setValue(0, 0, 1);

    int i;

    for(i = 0; i <= k_total + l; i++) {
        //fprintf(stderr, "Iteration: %d\n", i);
        Z.getCol(i + 1, &zCol);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z, 0.0, zCol);

        int a = i - l;
        if(i > 0) {
            MPI_Wait(&r2, MPI_STATUS_IGNORE);
            for (int j = 0; j <= i; j++) {
                //fprintf(stderr, "Setting G(%d, %d) to %f\n", j, i, dotProds[j]);
                G.setValue(j, i, dotProds[j]);
            }
        }

        if (a >= 0) {
            double sumH1, sumH2 = 0;

            for (int j = a - l + 2; j <= a; j++) {
                if(j < 0)
                    continue;
                sumH1 = 0;
                for (int k = 0; k <= j - 1; k++) {
                    //fprintf(stderr, "getting G(%d, %d) and G(%d, %d)\n", k, j, k, a + 1);
                    sumH1 += G.getValue(k, j) * G.getValue(k, a + 1);
                }
                //fprintf(stderr, "Getting G(%d, %d) ( = %f) \n", j, j, G.getValue(j, j));
                G.setValue(j, a + 1, (G.getValue(j, a + 1) - sumH1) / G.getValue(j, j));
                //fprintf(stderr, "Setting G(%d, %d) to %f\n", j, a + 1, G.getValue(j, a + 1));
            }


            for (int k = 0; k <= a; k++) {
                //fprintf(stderr, "getting G(%d, %d) ( = %f) \n", k, a + 1, G.getValue(k, a + 1));
                sumH2 += pow(G.getValue(k, a + 1), 2);
            }
            double gVal = G.getValue(a + 1, a + 1);
            //Check for breakdown and restart or reorthogonalize if necessary
            if (gVal - sumH2 < 0) {
                fprintf(stderr, "Breakdown at k = %d\n", i);
                break;
            }

            G.setValue(a + 1, a + 1, sqrt(gVal - sumH2));

            if (a < l) {
                for (int j = 0; j <= a; j++) {
                    sumH1 = 0;
                    for (int k = 0; k <= a - 1; k++) {
                        //fprintf(stderr, "getting H(%d, %d) and G(%d, %d)\n", j, k, k, a);
                        sumH1 += H->getValue(j, k) * G.getValue(k, a);
                    }
                    //fprintf(stderr, "getting G(%d, %d), G(%d, %d) and G(%d, %d)\n", j, a+1, j, a, a,a);
                    H->setValue(j, a, (G.getValue(j, a + 1) + G.getValue(j, a) - sumH1) / G.getValue(a, a));
                    //fprintf(stderr, "Setting H(%d, %d) to %f\n", j, a, (G.getValue(j, a + 1) + G.getValue(j, a) - sumH1) / G.getValue(a, a));
                }
                //fprintf(stderr, "getting G(%d, %d) and G(%d, %d)\n", a + 1, a + 1, a, a);
                H->setValue(a + 1, a, G.getValue(a + 1, a + 1) / G.getValue(a, a));
                //fprintf(stderr, "Setting H(%d, %d) to %f\n", a + 1, a, H->getValue(a + 1, a));
            }
            else {
                for (int j = 0; j <= a; j++) {
                    sumH1 = 0;
                    sumH2 = 0;
                    for (int k = 0; k <= a + 1 - l; k++) {
                        //fprintf(stderr, "getting G(%d, %d) and H(%d, %d)\n", j, k + l, k, a - l);
                        sumH1 += G.getValue(j, k + l) * H->getValue(k, a - l);
                    }
                    for (int k = j - 1; k <= a - 1; k++) {
                        //fprintf(stderr, "getting H(%d, %d) and G(%d, %d)\n", j, k, k, a);
                        sumH2 += H->getValue(j, k) * G.getValue(k, a);
                    }
                    //fprintf(stderr, "getting G(%d, %d)\n", a, a);
                    H->setValue(j, a, (sumH1 - sumH2) / G.getValue(a, a));
                    //fprintf(stderr, "Setting H(%d, %d) to %f\n", j, a, H->getValue(j, a));
                }
                //fprintf(stderr, "getting G(%d, %d), G(%d, %d) and H(%d, %d)\n", a + 1, a + 1, a, a, a + 1 - l, a - l);
                H->setValue(a + 1, a, (G.getValue(a + 1, a + 1) * H->getValue(a + 1 - l, a - l)) / G.getValue(a, a));
                //fprintf(stderr, "Setting H(%d, %d) to %f\n", a + 1, a, H->getValue(a + 1, a));

            }

            memset(vValVec, 0, counts[me] * sizeof(double));
            //#pragma omp for reduction(+:vValVec[:counts[me]]) private(vCol)
            for (int j = 0; j <= a - 1; j++) {
                V->getCol(j, &vCol);
                gVal = G.getValue(j, a);
                if(gVal == 0)
                    continue;
                //fprintf(stderr, "Getting G(%d, %d)\n", j, a);
                for (int idx = 0; idx < counts[me]; idx++) {
                    vValVec[idx] += vCol[idx] * gVal;
                }
            }

            V->getCol(a, &vCol);
            Z.getCol(a, &zCol);
            gVal = G.getValue(a, a);
            //fprintf(stderr, "Getting G(%d, %d) = %f\n", a, a, gVal);
            //fprintf(stderr, "gVal: %f\n", gVal);
            //#pragma omp for
            for (int idx = 0; idx < counts[me]; idx++) {
                vCol[idx] = (zCol[idx] - vValVec[idx]) / gVal;
            }
            if(a > 0) {
                memset(vValVec, 0, counts[me] * sizeof(double));
                //#pragma omp for reduction(+:vValVec[:counts[me]]) private(vCol)
                for (int j = 0; j <= a - 1; j++) {
                    Z.getCol(j + l, &zCol);
                    double hVal = H->getValue(j, a - 1);
                    if(hVal == 0)
                        continue;
                    //fprintf(stderr, "Getting H(%d, %d)\n", j, a - 1);
                    for (int idx = 0; idx < counts[me]; idx++) {
                        vValVec[idx] += zCol[idx] * hVal;
                    }
                }

                Z.getCol(i + 1, &zCol);
                //#pragma omp for
                double hVal = H->getValue(a, a - 1);
                //fprintf(stderr, "Getting H(%d, %d) = %f\n", a, a - 1, hVal);
                //fprintf(stderr, "hVal: %f\n", hVal);
                for (int idx = 0; idx < counts[me]; idx++) {
                    zCol[idx] = (zCol[idx] - vValVec[idx]) / hVal;
                }
            }
        }
        MPI_Iallgatherv(zCol, counts[me], MPI_DOUBLE, z, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &r1);

        memset(dotProds, 0, (i + 2) * sizeof(double));
        Z.getCol(i + 1, &zCol);
        for (int j = 0; j <= a; j++) {
            V->getCol(j, &vCol);
            dotProds[j] = cblas_ddot(counts[me], zCol, 1, vCol, 1);
            //fprintf(stderr, "1, j: %d, dotProds[j]: %f\n", j, dotProds[j]);
        }
        Z.getCol(i + 1, &zCol);
        for (int j = a + 1; j <= i + 1; j++) {
            if(j < 0)
                 continue;
            Z.getCol(j, &xZcol);
            dotProds[j] = cblas_ddot(counts[me], zCol, 1, xZcol, 1);
            //fprintf(stderr, "2, j: %d, dotProds[j]: %f\n", j, dotProds[j]);
        }

        fflush(stderr);

        MPI_Iallreduce(MPI_IN_PLACE, dotProds, i + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &r2);

        MPI_Wait(&r1, MPI_STATUS_IGNORE);

    }
    free(z);
    free(privZ);

    delete[] vValVec;
    delete[] dotProds;

    return i;
}