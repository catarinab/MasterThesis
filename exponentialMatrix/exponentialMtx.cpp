#include <omp.h>
#include <mpi.h>
#include <cstring>
#include <utility>

#include "../utils/headers/help_process.hpp"
#include "../utils/headers/distr_mtx_ops.hpp"
#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/pade_exp_approx.hpp"
#include "../utils/headers/arnoldiIteration.hpp"
#include "../utils/headers/io_ops.hpp"

using namespace std;


//Calculate the approximation of exp(A)*b
double getApproximation(dense_matrix V, const dense_matrix& expH, double betaVal, int krylovDegree) {
    dense_vector unitVec = dense_vector(krylovDegree);
    unitVec.insertValue(0, 1);

    if(betaVal != 1)
        V = V * betaVal;

    return vectorTwoNorm(denseMatrixMult(V, expH).getCol(0));
}

//Process input arguments
void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxName, double * normVal) {
    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxName = argv[i+1];
        }
        else if(strcmp(argv[i], "-n") == 0) {
            *normVal = stod(argv[i+1]);
        }
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double exec_time_pade, exec_time_arnoldi, exec_time;

    //input values
    int krylovDegree; 
    string mtxPath;
    double normVal;
    processArgs(argc, argv, &krylovDegree, &mtxPath, &normVal);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int size = (int) readHeader(mtxPath).first;
    initGatherVars(size, nprocs);

    //initializations of needed matrix and vectors
    csr_matrix A = buildPartMatrix(mtxPath, me, displs, counts);

    dense_vector b(size);
    b.getOnesVec();
    b = b / b.getNorm2();
    double betaVal = b.getNorm2();

    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    exec_time_arnoldi = -omp_get_wtime();
    arnoldiIteration(A, b, krylovDegree, size, me, nprocs, &V, &H, 1);
    exec_time_arnoldi += omp_get_wtime();
    //root node performs pade approximation and outputs results
    if(me == 0) {
        exec_time_pade = -omp_get_wtime();
        dense_matrix expH = scalingAndSquaring(H);
        exec_time_pade += omp_get_wtime();

        double resNorm = getApproximation(V, expH, betaVal, krylovDegree);

        exec_time += omp_get_wtime();

        //output results
        printf("exec_time_arnoldi: %f\n", exec_time_arnoldi);
        printf("exec_time_pade: %f\n", exec_time_pade);
        printf("diff: %.15f\n", abs(normVal - resNorm));
        printf("2Norm: %.15f\n", resNorm);
        printf("exec_time: %f\n", exec_time);

    }

    mkl_sparse_destroy(A.getMKLSparseMatrix());
    free(displs);
    free(counts);

    MPI_Finalize();

    return 0;
}