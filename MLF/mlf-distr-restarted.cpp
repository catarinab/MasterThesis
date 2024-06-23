#include <iostream>
#include <omp.h>
#include <cstring>
#include <mpi.h>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/calculate_MLF.hpp"
#include "../utils/headers/arnoldi_iteration.hpp"
#include "../utils/headers/io_ops.hpp"
#include "../utils/headers/distr_mtx_ops.hpp"

using namespace std;

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
    csr_matrix A = buildPartialMatrix(mtxPath, me, displs, counts);

    dense_vector b = dense_vector(size);
    b.insertValue(0, 1);
    double betaNormB = b.getNorm2();

    dense_matrix V(counts[me], krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time_arnoldi = -omp_get_wtime();
    double norm =  restartedArnoldiIteration_MLF(A, b, krylovDegree, size, me, &V, &H, alpha, beta);
    exec_time_arnoldi += omp_get_wtime();
    if(me == 0) {
        cout << exec_time_arnoldi << "," << 0 << "," << exec_time_arnoldi << endl;
        cerr << norm << endl;
    }

    free(displs);
    free(counts);
    mkl_sparse_destroy(A.getMKLSparseMatrix());

    MPI_Finalize();


    return 0;
}