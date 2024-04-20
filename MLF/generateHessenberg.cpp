#include <iostream>
#include <string>
#include <omp.h>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/arnoldiIteration-shared.hpp"
#include "../utils/headers/mmio.h"

using namespace std;


int main (int argc, char* argv[]) {
    //input values

    double exec_time;

    if(argc != 3){
        cerr << "Usage: " << argv[0] << " <init-krylov-degree> <end-krylov-degree> " << endl;
        return 1;
    }

    int initKrylovDegree = stoi(argv[1]);
    int endKrylovDegree = stoi(argv[2]);
    cout << "krylov degree: " << initKrylovDegree << endl;
    cout << "endKrylovDegree" << endKrylovDegree << endl;
    string mtxPath;
    mtxPath = "A.mtx";

    cout << mtxPath << endl;

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMtx(mtxPath);
    int size = (int) A.getSize();

    dense_vector b(size);
    b.getOnesVec();
    b = b / b.getNorm2();
    int totalIter = 0;

    for(int krylovDegree = initKrylovDegree; krylovDegree <= endKrylovDegree; krylovDegree +=20)
        totalIter += krylovDegree;

    int stat = mkl_sparse_set_mv_hint(A.getMKLSparseMatrix(),SPARSE_OPERATION_NON_TRANSPOSE,A.getMKLDescription(),
                                      totalIter);

    if (stat != SPARSE_STATUS_SUCCESS) {
        cout << stat << endl;
        cerr << "Error in mkl_sparse_set_mv_hint" << endl;
        return 1;
    }

    stat = mkl_sparse_optimize(A.getMKLSparseMatrix());

    if (stat != SPARSE_STATUS_SUCCESS) {
        cerr << "Error in mkl_sparse_optimize" << endl;
        return 1;
    }

    for(int krylovDegree = initKrylovDegree; krylovDegree <= endKrylovDegree; krylovDegree +=20) {
        cout << "Doing Krylov" << krylovDegree << endl;
        dense_matrix V(size, krylovDegree);
        dense_matrix H(krylovDegree, krylovDegree);

        cout << "starting arnoldi iteration..." << endl;

        exec_time = -omp_get_wtime();

        arnoldiIteration(A, b, krylovDegree, size, &V, &H);

        exec_time += omp_get_wtime();

        printf("exec_time: %f\n", exec_time);

        int nz = krylovDegree * (krylovDegree + 1) / 2 + (krylovDegree - 1);

        MM_typecode matcode;

        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);

        FILE *mtxmkt;

        string fileName = "krylov/" + to_string(krylovDegree) + ".mtx";

        mtxmkt = fopen(fileName.c_str(), "w");

        mm_write_banner(mtxmkt, matcode);
        mm_write_mtx_crd_size(mtxmkt, krylovDegree, krylovDegree, nz);


         /*NOTE: matrix market files use 1-based indices, i.e. first element
          of a vector has index 1, not 0.*/

        for (int row = 0; row < krylovDegree; row++) {
            for (int col = 0; col < krylovDegree; col++) {
                if (row <= col + 1) {
                    fprintf(mtxmkt, "%d %d %.16e\n", row + 1, col + 1, H.getValue(row, col));
                }
            }
        }

        fclose(mtxmkt);

        H.printVector("krylov/" + to_string(krylovDegree) + "-vector.txt");
    }

    mkl_sparse_destroy(A.getMKLSparseMatrix());

    return 0;
}