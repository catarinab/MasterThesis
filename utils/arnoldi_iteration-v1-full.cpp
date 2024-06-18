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

    //fprintf(stderr, "Arnoldi iteration, me: %d\n", me);
    double dotProd = 0;
    double wNorm = 0;
    double ddot;
    double * w;
    double * newVCol;
    MPI_Comm new_comm;
    MPI_Request wGather, normReduce, dotReduce;
    auto * dotProds = (double *) malloc(k_total * sizeof(double));
    if(me == 0) {
         w = (double *) malloc(m * sizeof(double));
         newVCol = (double *) malloc(m * sizeof(double));
         MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, me, &new_comm);
    }
    else {
        MPI_Comm_split(MPI_COMM_WORLD, 1, me, &new_comm);
    }

    //auxiliary
    auto * privW = (double *) malloc(counts[me] * sizeof(double));
    V->setCol(0, initVec);
    double * vCol;
    V->getCol(0, &vCol);

    for(int k = 1; k < k_total + 1; k++) {
        if (me == 0) {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                            vCol, 0.0, w);
            cout << endl;
        }

        MPI_Scatterv(w, counts, displs, MPI_DOUBLE, privW, counts[me], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        if(me == 0 && k>1) {
            fprintf(stderr, "Before Wait, me: %d, k: %d\n", me, k);
            MPI_Wait(&dotReduce, MPI_STATUS_IGNORE);
            fprintf(stderr, "After Wait, me: %d, k: %d\n", me, k);
            H->setCol(k - 2, dotProds, k - 2);
            H->setValue(k - 1, k - 2, wNorm);
            wNorm = 0;
            memset(dotProds, 0, k * sizeof(double));
        }

        else if(me != 0) {
            for (int j = 0; j < k; j++) {
                V->getCol(j, &vCol, displs[me]);

                ddot = cblas_ddot(counts[me], privW, 1, vCol, 1);

                fprintf(stderr, "me: %d, k: %d, j: %d, ddot: %f\n", me, k, j, ddot);

                MPI_Allreduce(&ddot, &dotProd, 1, MPI_DOUBLE, MPI_SUM, new_comm);
                dotProds[j] = dotProd;

                fprintf(stderr, "me: %d, k: %d, j: %d, dotProd: %f\n", me, k, j, dotProd);

                cblas_daxpy(counts[me], -dotProd, vCol, 1, privW, 1);
            }
            wNorm = cblas_dnrm2(counts[me], privW, 1.0);
        }

        if(k == k_total) break;

        MPI_Allreduce(MPI_IN_PLACE, &wNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(wNorm < EPS52) break;

        #pragma omp parallel for
        for (int i = 0; i < counts[me]; i++) {
            privW[i] /= wNorm;
        }

        fprintf(stderr, "Before gather, me: %d, k: %d\n", me, k);

        V->getCol(k, &vCol);

        MPI_Gatherv(privW, counts[me], MPI_DOUBLE, vCol, counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        fprintf(stderr, "After gather, me: %d, k: %d\n", me, k);

        MPI_Reduce(MPI_IN_PLACE, dotProds, k, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

        fprintf(stderr, "After ireduce, me: %d, k: %d\n", me, k);


    }

    if (me != 0) {
        MPI_Comm_free(&new_comm);
    }
    else {
        H->printMatrix();
        free(w);
        free(newVCol);
    }
    free(privW);
    free(dotProds);

    return k_total;
}