#include <iostream>
#include <omp.h>
#include <cstring>
#include <mpi.h>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/calculate_MLF.hpp"
#include "../utils/headers/arnoldi_iteration-v1.hpp"
#include "../utils/headers/io_ops.hpp"
#include "../utils/headers/distr_mtx_ops.hpp"

using namespace std;

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix V, const dense_matrix& mlfH, double betaVal) {

    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

//Process input arguments
void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxName) {
    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxName = argv[i+1];
        }
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double exec_time_schur, exec_time_arnoldi, exec_time;

    //input values
    double alpha = 0.5;
    double beta = 0;

    /*cerr << "mkl max threads: " << mkl_get_max_threads() << endl;
    cerr << "omp max threads: " << omp_get_max_threads() << endl;*/

    int krylovDegree = 3;
    string mtxPath = "A.mtx";
    processArgs(argc, argv, &krylovDegree, &mtxPath);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    fprintf(stderr, "Me: %d, omp max threads: %d, mkl max threads: %d\n", me, omp_get_max_threads(), mkl_get_max_threads());

    //initializations of needed matrix and vectors
    int size = (int) readHeader(mtxPath).first;
    initGatherVars(size, nprocs);

    //initializations of needed matrix and vectors
    csr_matrix A = buildPartMatrix(mtxPath, me, displs, counts);

    dense_vector b = dense_vector(size);
    b.insertValue(0, 1);
    //b.insertValue(floor(size/2), 1);
    double betaVal = b.getNorm2();

    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    exec_time_arnoldi = -omp_get_wtime();
    arnoldiIteration(A, b, krylovDegree, size, me, &V, &H);
    exec_time_arnoldi += omp_get_wtime();

    if(me == 0){
        exec_time_schur = -omp_get_wtime();
        dense_matrix mlfH = calculate_MLF((double *) H.getDataPointer(), alpha, beta, krylovDegree);
        exec_time_schur += omp_get_wtime();

        dense_vector res = getApproximation(V, mlfH, betaVal);

        exec_time += omp_get_wtime();

        cout << exec_time_arnoldi << "," << exec_time_schur << "," << exec_time << endl;
        cerr << res.getNorm2() << endl;
    }

    mkl_sparse_destroy(A.getMKLSparseMatrix());
    free(displs);
    free(counts);

    MPI_Finalize();


    return 0;
}