#include "headers/arnoldiIteration-shared.hpp"
#include <omp.h>

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
int arnoldiIteration(const csr_matrix& A, dense_vector& initVec, int k_total, int m, dense_matrix * V,
                     dense_matrix * H) {

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

    V->setCol(0, initVec);

    int k;

    //auxiliary
    auto * w = new double[m];
    double * vCol;

    double exec_time_mv = 0;
    double exec_time_dotProd = 0;
    double exec_time_axpy = 0;
    double exec_time_norm = 0;
    double exec_time_setValue = 0;
    double exec_time_setHValue = 0;
    double tempTime = 0;

    V->getCol(0, &vCol);

    for(k = 1; k < k_total + 1; k++) {
        tempTime = -omp_get_wtime();
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, w);
        exec_time_mv += tempTime + omp_get_wtime();

        for(int j = 0; j < k; j++) {
            V->getCol(j, &vCol);

            //dotprod between w and V->getCol(j)
            tempTime = -omp_get_wtime();
            double dotProd = cblas_ddot(m, w, 1, vCol, 1);
            exec_time_dotProd += tempTime + omp_get_wtime();

            // w = w - dotProd[j] * V(:, j)
            tempTime = -omp_get_wtime();
            cblas_daxpy(m, -dotProd, vCol, 1, w, 1);
            exec_time_axpy += tempTime + omp_get_wtime();

            H->setValue(j, k - 1, dotProd);
        }

        if(k < k_total){
            V->getCol(k, &vCol);
            double tempNorm = 0;
            #pragma omp parallel
            {
                #pragma omp single
                    tempTime = -omp_get_wtime();
                #pragma omp for reduction(+:tempNorm)
                for(int i = 0; i < m; i++){
                    tempNorm += w[i] * w[i];
                }
                #pragma omp single
                    exec_time_norm += tempTime + omp_get_wtime();
                double wNorm = sqrt(tempNorm);
                if(wNorm != 0)
                    #pragma omp single
                    {
                        tempTime = -omp_get_wtime();
                    }
                    #pragma omp for nowait
                    for(int i = 0; i < m; i++){
                        vCol[i] = w[i] / wNorm;
                    }
                    #pragma omp single
                    {
                        exec_time_setValue += tempTime + omp_get_wtime();
                    }
                #pragma omp single
                {
                    H->setValue(k, k - 1, wNorm);
                }
            }

        }

    }

    cout << "mv: " << exec_time_mv << endl;
    cout << "dotProd: " << exec_time_dotProd << endl;
    cout << "axpy: " << exec_time_axpy << endl;
    cout << "norm: " << exec_time_norm << endl;
    cout << "setValue: " << exec_time_setValue << endl;

    delete[] w;

    return k;
}