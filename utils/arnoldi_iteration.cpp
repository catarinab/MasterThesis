#include <mpi.h>
#include <omp.h>

#include "headers/arnoldi_iteration.hpp"
#include "headers/distr_mtx_ops.hpp"
#include "headers/help_process.hpp"

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

    int func = 0;
    int sendEnd = ENDTAG;

    int stat = mkl_sparse_set_mv_hint(A.getMKLSparseMatrix(),SPARSE_OPERATION_NON_TRANSPOSE,A.getMKLDescription(),
                                      k_total);

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_set_mv_hint" << endl;
        return 1;
    }

    stat = mkl_sparse_optimize(A.getMKLSparseMatrix());

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_optimize" << endl;
        return 1;
    }

    //helper nodes
    if(me != 0)
        return helpProcess(A, me, m, func, displs, counts);

    V->setCol(0, initVec);

    int k;

    //auxiliary
    dense_vector b(m);
    dense_vector w(m);
    double * vCol;

    double temp_exec_time;

    for(k = 1; k < k_total + 1; k++) {

        cout << "iteration k " << endl;
        V->getCol(k-1, &b);
        temp_exec_time = -omp_get_wtime();
        distrMatrixVec(A, b, w, m);
        temp_exec_time += omp_get_wtime();

        cout << "distr matrix vec: " << temp_exec_time << endl;

        for(int j = 0; j < k; j++) {
            V->getCol(j, &b);
            temp_exec_time = -omp_get_wtime();
            double dotProd = distrDotProduct(w, b, m, me);
            temp_exec_time += omp_get_wtime();
            cout << "distr dot product: " << temp_exec_time << endl;

            temp_exec_time = -omp_get_wtime();
            distrSumOp(w, b, -dotProd, m, me);
            temp_exec_time += omp_get_wtime();
            cout << "distr axpy: " << temp_exec_time << endl;
            H->setValue(j, k-1, dotProd);
        }

        if(k == k_total) break;

        double wNorm = w.getNorm2();

        if(wNorm != 0) {
            H->setValue(k, k - 1, wNorm);
            V->getCol(k, &vCol);
            temp_exec_time = -omp_get_wtime();
            #pragma omp parallel for
            for(int i = 0; i < m; i++){
                vCol[i] = w.values[i] / wNorm;
            }
            temp_exec_time += omp_get_wtime();
            cout << "vCol change: " << temp_exec_time << endl;
        }
    }
    MPI_Bcast(&sendEnd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return k;
}