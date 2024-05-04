#include <iostream>
#include <mkl.h>
#include <cstring>

#include "../utils/headers/dense_matrix.hpp"
#include "../utils/headers/calculate-MLF.hpp"
#include "../utils/headers/schur-blocking.hpp"
#include "../utils/headers/arnoldiIteration-shared-nu.hpp"
#include "../utils/headers/mtx_ops_mkl.hpp"

using namespace std;

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

    int krylovDegree;
    string mtxPath = "mtx/784-convect.mtx";
    double normVal = 0;
    processArgs(argc, argv, &krylovDegree, &mtxPath, &normVal);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMtx(mtxPath);
    int size = (int) A.getSize();

    dense_vector b(size);
    b.getOnesVec();
    b = b / b.getNorm2();

    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    arnoldiIteration(A, b, krylovDegree, size, &V, &H, 3);

    H.printMatrix();
    return 0;
}