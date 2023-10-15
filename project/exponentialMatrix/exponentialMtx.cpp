#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "../utils/distr_mtx_ops.cpp"
#include "../utils/helpProccess.cpp"
#include "../utils/dense_Matrix.cpp"

using namespace std;
#define epsilon 0.000000000001 //10^-12

bool debugMtr = false;
bool vecFile = false;
int beta = 1;

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
    int helpSize = 0;
    int sendEnd = ENDTAG;

    while(me != 0) {
        MPI_Bcast(&helpSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(helpSize == ENDTAG) return 0;
        else if(helpSize > 0)
            helpProccess(A, me, m, helpSize, nprocs, displs, counts);
    }

    b = b / b.getNorm2();
    V->setCol(0, b);

    int k = 1;

    //auxiliar
    Vector opResult(m);

    for(k = 1; k < n + 1; k++) {
        cout << "k: " << k << endl;
        cout << V->getCol(k-1).size << endl;
        cout << A.getSize() << endl;
        Vector w = distrMatrixVec(A, V->getCol(k-1), m, me, nprocs);
        for(int j = 0; j <= k; j++) {
            H->setValue(j, k-1, distrDotProduct(w, V->getCol(j), m, me, nprocs));
            opResult = V->getCol(j) * H->getValue(j, k-1);
            w = distrSubOp(w, opResult, m, me, nprocs);
        }
        H->setValue(k, k - 1, w.getNorm2());
        if(H->getValue(k, k - 1) > epsilon)
            V->setCol(k, w / H->getValue(k, k - 1));

        else{
            printf("Krylov subspace exhausted at iteration %d.", k);
            MPI_Bcast(&sendEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return k;
        }

    }
    MPI_Bcast(&sendEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return --k;
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double exec_time;
    bool vecFile = false;

    int krylovDegree = 3;
    int finalKrylovDegree;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx("/home/cat/uni/thesis/project/mtx/Trefethen_20b/Trefethen_20b.mtx");
    int size = csr.getSize();

    Vector b(size);

    b.getOnesVec();

    initGatherVars(size, nprocs);


    dense_Matrix V(size, krylovDegree + 1);
    dense_Matrix H(krylovDegree + 1, krylovDegree);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    //from this, we get the Orthonormal basis of the Krylov subspace (V) and the upper Hessenberg matrix (H)
    finalKrylovDegree = arnoldiIteration(csr, b, krylovDegree, size, me, nprocs, &V, &H);

    //e^A = V * e^H * V^T

    MPI_Barrier(MPI_COMM_WORLD);

    exec_time += omp_get_wtime();

    if(me == 0) {
        V.printAttr("V");
        H.printAttr("H");
        cout << "finalKrylovDegree: " << finalKrylovDegree << endl;
        cout << "exec_time: " << exec_time << endl;
    }

    free(displs);
    free(counts);
    MPI_Finalize();
    return 0;
}