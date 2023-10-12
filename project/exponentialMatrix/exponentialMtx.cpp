#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "../../utils/distr_mtx_ops.cpp"
#include "../../utils/helpProccess.cpp"
#include "../../utils/dense_Matrix.cpp"

using namespace std;
#define epsilon 0.00000000000001 
#define N 100

bool debugMtr = false;
bool vecFile = false;
int maxIter;

/*Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : An m × m array. (CSR_Matrix)
    b : Initial mx1 (Vector).
    n : One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1 (int)
    
    Returns
    -------
    Q : An m x (n + 1) array (dense_matrix rows:m cols: n+1), where the columns are an orthonormal basis of the Krylov subspace.
    H : An (n + 1) x n array (dense_matrix rows: n+1, cols: n). A on basis Q. It is upper Hessenberg.*/
int arnoldiIteration(CSR_Matrix A, Vector b, int n, int m, int me, int nprocs, dense_Matrix * V, dense_Matrix * H) {

    b = b / b.getNorm2();
    V->setCol(0, b);

    int k = 1;

    //auxiliar
    Vector opResult(m);

    for(k = 1; k < n; k++) {
        Vector w = distrMatrixVec(A, V->getCol(k-1), m, me, nprocs);
        for(int j = 0; j <= k; j++) {
            H->setValue(j, k, distrDotProduct(w, V->getCol(j), m, me, nprocs));
            opResult = distrDotProduct(H->getValue(j, k-1), V->getCol(j), m, me, nprocs);
            w = distrSubOp(w, opResult, m, me, nprocs);
        }
        H->setValue(k, k - 1, w.getNorm2());
        if(H->getValue(k, k - 1) > epsilon) 
            V->setCol(k, w / H->getValue(k, k - 1));

        else 
            return k;
    }
    return k;
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    int finalIterArnoldi;
    double exec_time;
    string input_fileVec;
    bool vecFile = false;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx("/home/cat/uni/thesis/project/mtx/ted_B_unscaled/ted_B_unscaled.mtx");
    int size = csr.getSize();

    Vector b(size);

    if(vecFile) {
        b.setValues(readFile_vec("/home/cat/uni/thesis/project/Vec/vec_ted_b", size));
    } else {
        b.getRandomVec();
    }

    initGatherVars(size, nprocs);

    dense_Matrix V(size, N + 1);
    dense_Matrix H(N + 1, N);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();

    //from this, we get the Orthonormal basis of the Krylov subspace (V) and the upper Hessenberg matrix (H)
    finalIterArnoldi = arnoldiIteration(csr, b, N, size, me, nprocs, &V, &H);

    //e^A = V * e^H * V^T

    MPI_Barrier(MPI_COMM_WORLD);

    exec_time += omp_get_wtime();

    free(displs);
    free(counts);
    MPI_Finalize();
    return 0;
}