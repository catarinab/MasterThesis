#include <iostream>
#include <omp.h>
#include <cstring>

#include "../utils/headers/mtx_ops_mkl.hpp"

using namespace std;

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix V, const dense_matrix& mlfH, double betaVal) {

    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

//Process input arguments
void processArgs(int argc, char* argv[], string * mtxName) {
    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-m") == 0) {
            *mtxName = argv[i+1];
        }
    }
}


int main (int argc, char* argv[]) {

    int alpha = 1;
    int beta = 1;


    cerr << "mkl max threads: " << mkl_get_max_threads() << endl;
    cerr << "omp max threads: " << omp_get_max_threads() << endl;

    string mtxPath = "lamp-0.18";
    processArgs(argc, argv, &mtxPath);

    //initializations of needed matrix and vectors
    csr_matrix C = buildFullMatrix(mtxPath + "-conv.mtx");
    int size = (int) C.getSize();

    csr_matrix D = buildFullMatrix(mtxPath + "-diff.mtx");

    csr_matrix B = csr_matrix();

    mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, C.getMKLSparseMatrix(), 1,
                     D.getMKLSparseMatrix(),
                     B.getMKLSparseMatrixPointer());

    mkl_sparse_destroy(C.getMKLSparseMatrix());
    mkl_sparse_destroy(D.getMKLSparseMatrix());

    csr_matrix M = buildInverseDiagonalMatrix(mtxPath + "-mass.vec");

    //A = - M^-1 B

    B.convertInternal(size);

    //B.printAttr();

    csr_matrix A = csr_matrix();
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, M.getMKLSparseMatrix(), B.getMKLSparseMatrix(),
                    A.getMKLSparseMatrixPointer());

    A.convertInternal(size);

    //A.printAttr();


    A.saveMatrixMarketFile((string &) "A-lamp-0.18.mtx");

    mkl_sparse_destroy(B.getMKLSparseMatrix());
    mkl_sparse_destroy(M.getMKLSparseMatrix());
    mkl_sparse_destroy(A.getMKLSparseMatrix());


    return 0;
}