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
void getApproximation(dense_matrix& V, const dense_matrix& mlfH, double betaVal, dense_vector & res) {
    if(betaVal != 1)
        V = V * betaVal;

    denseMatrixMult(V, mlfH).getCol(0, &res);
}

dense_vector restartedArnoldiIteration_MLF(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                                           dense_matrix * H, double alpha, double beta, double betaNormB, int l) {
    int k = 0;
    dense_vector b(m);
    dense_vector temp(counts[me]);
    dense_vector res(counts[me]);


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
        k += size;

        MPI_Bcast(mlfH.getValues(), size * size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        getApproximation(*V, mlfH, betaNormB, res);

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
    double * vCol, *xZcol;
    double * zCol;
    auto vValVec = new double[counts[me]]();
    auto dotProds = new double[k_total + l + 2];
    MPI_Request r1, r2;

    V->setCol(0, initVec, displs[me], counts[me]);
    Z.setCol(0, initVec, displs[me], counts[me]);
    G.setValue(0, 0, 1);

    int i;

    for(i = 0; i <= k_total + l; i++) {
        Z.getCol(i + 1, &zCol);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        z, 0.0, zCol);

        int a = i - l;
        if(i > 0) {
            MPI_Wait(&r2, MPI_STATUS_IGNORE);
            for (int j = 0; j <= i; j++) {
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
                    sumH1 += G.getValue(k, j) * G.getValue(k, a + 1);
                }
                G.setValue(j, a + 1, (G.getValue(j, a + 1) - sumH1) / G.getValue(j, j));
            }


            for (int k = 0; k <= a; k++) {
                sumH2 += pow(G.getValue(k, a + 1), 2);
            }
            double gVal = G.getValue(a + 1, a + 1);

            //Check for breakdown and restart or reorthogonalize if necessary
            if (gVal - sumH2 < 0) {
                break;
            }

            G.setValue(a + 1, a + 1, sqrt(gVal - sumH2));

            //porque ???
            Z.getCol(a, &zCol);
            #pragma omp parallel private(vCol, gVal) firstprivate(zCol) shared(vValVec, a, i)
            {
                if (a < l) {
                    #pragma omp for private(sumH1)
                    for (int j = 0; j <= a; j++) {
                        sumH1 = 0;
                        for (int k = 0; k <= a - 1; k++) {
                            sumH1 += H->getValue(j, k) * G.getValue(k, a);
                        }
                        H->setValue(j, a, (G.getValue(j, a + 1) + G.getValue(j, a) - sumH1) / G.getValue(a, a));
                    }
                    #pragma omp single
                        H->setValue(a + 1, a, G.getValue(a + 1, a + 1) / G.getValue(a, a));
                } else {
                    #pragma omp for private(sumH1, sumH2)
                    for (int j = 0; j <= a; j++) {
                        sumH1 = 0;
                        sumH2 = 0;
                        for (int k = 0; k <= a + 1 - l; k++) {
                            sumH1 += G.getValue(j, k + l) * H->getValue(k, a - l);
                        }
                        for (int k = j - 1; k <= a - 1; k++) {
                            sumH2 += H->getValue(j, k) * G.getValue(k, a);
                        }
                        H->setValue(j, a, (sumH1 - sumH2) / G.getValue(a, a));
                    }
                    #pragma omp single
                        H->setValue(a + 1, a,
                                (G.getValue(a + 1, a + 1) * H->getValue(a + 1 - l, a - l)) / G.getValue(a, a));

                }

                #pragma omp for reduction(+:vValVec[:counts[me]]) private(vCol, gVal)
                for (int j = 0; j <= a - 1; j++) {
                    V->getCol(j, &vCol);
                    gVal = G.getValue(j, a);
                    if(gVal == 0)
                        continue;
                    for (int idx = 0; idx < counts[me]; idx++) {
                        vValVec[idx] += vCol[idx] * gVal;
                    }
                }

                V->getCol(a, &vCol);
                gVal = G.getValue(a, a);
                #pragma omp for
                for (int idx = 0; idx < counts[me]; idx++) {
                    vCol[idx] = (zCol[idx] - vValVec[idx]) / gVal;
                    vValVec[idx] = 0;
                }

                if(a > 0) {
                    #pragma omp for reduction(+:vValVec[:counts[me]]) private(zCol)
                    for (int j = 0; j <= a - 1; j++) {
                        Z.getCol(j + l, &zCol);
                        double hVal = H->getValue(j, a - 1);
                        if(hVal == 0)
                            continue;
                        for (int idx = 0; idx < counts[me]; idx++) {
                            vValVec[idx] += zCol[idx] * hVal;
                        }
                    }

                    Z.getCol(i + 1, &zCol);
                    double hVal = H->getValue(a, a - 1);
                    #pragma omp for
                    for (int idx = 0; idx < counts[me]; idx++) {
                        zCol[idx] = (zCol[idx] - vValVec[idx]) / hVal;
                        vValVec[idx] = 0;
                    }
                }
            }
        }


        MPI_Iallgatherv(zCol, counts[me], MPI_DOUBLE, z, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &r1);

        memset(dotProds, 0, (i + 2) * sizeof(double));
        Z.getCol(i + 1, &zCol);
        for (int j = 0; j <= a; j++) {
            V->getCol(j, &vCol);
            dotProds[j] = cblas_ddot(counts[me], zCol, 1, vCol, 1);
        }
        Z.getCol(i + 1, &zCol);
        for (int j = a + 1; j <= i + 1; j++) {
            if(j < 0)
                 continue;
            Z.getCol(j, &xZcol);
            dotProds[j] = cblas_ddot(counts[me], zCol, 1, xZcol, 1);
        }

        MPI_Iallreduce(MPI_IN_PLACE, dotProds, i + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &r2);

        MPI_Wait(&r1, MPI_STATUS_IGNORE);

    }
    free(z);
    free(privZ);

    delete[] vValVec;
    delete[] dotProds;

    return i;
}