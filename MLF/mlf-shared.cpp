#include <iostream>
#include <omp.h>
#include <cstring>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/calculate-MLF.hpp"
#include "../utils/headers/arnoldiIteration-shared.hpp"

using namespace std;


//Calculate the approximation of exp(A)*b
double getApproximation(dense_matrix V, const dense_matrix& expH, double betaVal) {

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
    double exec_time_schur, exec_time_arnoldi, exec_time;

    //input values
    double alpha = 0.9228730416043215;
    double beta = 0;

    int krylovDegree;
    string mtxPath = "A-500.mtx";
    double normVal = 12.63377767489511;
    processArgs(argc, argv, &krylovDegree, &mtxPath, &normVal);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMtx(mtxPath);
    int size = (int) A.getSize();

    dense_vector b(size);
    b.getOnesVec();
    b = b / b.getNorm2();
    //b.insertValue(floor(size/2), 1);
    double betaVal = b.getNorm2();

    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    exec_time = -omp_get_wtime();
    exec_time_arnoldi = -omp_get_wtime();
    arnoldiIteration(A, b, krylovDegree, size, &V, &H);
    exec_time_arnoldi += omp_get_wtime();

    exec_time_schur = -omp_get_wtime();
    dense_matrix mlfH = calculate_MLF((double *) H.getDataPointer(), alpha, beta, krylovDegree);
    exec_time_schur += omp_get_wtime();

    double resNorm = getApproximation(V, mlfH, betaVal);

    //cerr << resNorm << endl;

    exec_time += omp_get_wtime();

    //output results
    cout << exec_time_schur << "," << (double) (abs(normVal - resNorm) / abs(normVal)) << endl;
    /*printf("exec_time_arnoldi: %f\n", exec_time_arnoldi);
    printf("exec_time_pade: %f\n", exec_time_schur);
    printf("diff: %.15f\n", abs(normVal - resNorm));
    printf("2Norm: %.15f\n", resNorm);
    printf("exec_time: %f\n", exec_time);*/
    
    mkl_sparse_destroy(A.getMKLSparseMatrix());


    return 0;
}