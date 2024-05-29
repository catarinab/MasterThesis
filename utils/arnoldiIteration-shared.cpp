#include <cstring>
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
    double tempTime = -omp_get_wtime();
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
    cout << "Hint and optimize: " << tempTime + omp_get_wtime() << endl;

    V->setCol(0, initVec);

    int k;

    //auxiliary
    auto* w = static_cast<double *>(aligned_alloc(64, m * sizeof(double)));
    double *vCol;
    auto * dotProd = new double[k_total + 1]();

    double exec_time_mv = 0;
    double exec_time_dotProd = 0;
    double exec_time_axpy = 0;
    double exec_time_norm = 0;
    double exec_time_setValue = 0;
    double exec_time_setHValue = 0;

    for(k = 1; k < k_total + 1; k++) {
        double tempNorm = 0;
        V->getCol(k-1, &vCol);
        memset(dotProd, 0, k * sizeof(double));

        tempTime = -omp_get_wtime();
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.getMKLSparseMatrix(), A.getMKLDescription(),
                        vCol, 0.0, w);
        exec_time_mv += tempTime + omp_get_wtime();


        #pragma omp parallel shared(dotProd) private(vCol) firstprivate(w)
        {
            for(int j = 0; j < k; j++) {
                V->getCol(j, &vCol);

                double localDotProd = 0;
                #pragma omp single
                    tempTime = -omp_get_wtime();
                //dotprod between w and V->getCol(j)
                #pragma omp for nowait
                for (int i = 0; i < m; i++) {
                    localDotProd += (w[i] * vCol[i]);
                }
                #pragma omp critical
                {
                    dotProd[j] += localDotProd;
                }
                #pragma omp barrier
                #pragma omp single
                {
                    exec_time_dotProd += tempTime + omp_get_wtime();
                    tempTime = -omp_get_wtime();
                }

                #pragma omp for nowait
                for(int i = 0; i < m; i++) {
                    w[i] = w[i] - vCol[i] * dotProd[j];
                }

                #pragma omp single
                    exec_time_axpy += tempTime + omp_get_wtime();
            }

            if(k < k_total) {
                #pragma omp single
                    tempTime = -omp_get_wtime();
                #pragma omp for reduction(+:tempNorm)
                for(int i = 0; i < m; i++) {
                    tempNorm += w[i] * w[i];
                }
                double wNorm = sqrt(tempNorm);
                #pragma omp single
                    exec_time_norm += tempTime + omp_get_wtime();
                if(wNorm != 0) {
                    //V(:, k) = w / wNorm
                    V->getCol(k, &vCol);
                    #pragma omp single
                        tempTime = -omp_get_wtime();
                    #pragma omp for nowait
                    for (int i = 0; i < m; i++) {
                        vCol[i] = w[i] / wNorm;
                    }

                    #pragma omp single
                    {
                        exec_time_setValue += tempTime + omp_get_wtime();
                        tempTime = -omp_get_wtime();
                    }

                    //H(k, k-1) = wNorm

                    #pragma omp for nowait
                    for (int i = 0; i < k; i++) {
                        H->setValue(i, k - 1, dotProd[i]);
                    }
                    #pragma omp single
                    {
                        H->setValue(k, k - 1, wNorm);
                    }
                    #pragma omp single
                        exec_time_setHValue += tempTime + omp_get_wtime();
                }
            }
        }
        //H(:, k-1) = dotProds
        for (int i = 0; i < k; i++) {
            H->setValue(i, k - 1, dotProd[i]);
        }
    }

    cout << "mv: " << exec_time_mv << endl;
    cout << "dotProd: " << exec_time_dotProd << endl;
    cout << "axpy: " << exec_time_axpy << endl;
    cout << "norm: " << exec_time_norm << endl;
    cout << "setValue: " << exec_time_setValue << endl;
    cout << "setHValue: " << exec_time_setHValue << endl;

    delete[] dotProd;

    free(w);

    return k;
}