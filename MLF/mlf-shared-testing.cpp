#include <iostream>
#include <omp.h>
#include <cstring>
#include <fstream>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/calculate-MLF.hpp"
#include "../utils/headers/arnoldiIteration-shared.hpp"

using namespace std;

dense_vector juliares;

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix V, const dense_matrix& mlfH, double betaVal) {

    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

void readJuliaVec(const string& filename = "juliares.txt") {
    ifstream inputFile(filename);
    if (inputFile) {
        double value;
        while (inputFile >> value) {
            juliares.values.push_back(value);
        }
        inputFile.close();
    }
    else {
        cout << "Error opening julia vector file" << endl;
    }
}

void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxPath, string * juliaPath) {
    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxPath = "A-" + std::string(argv[i + 1]) + ".mtx";
            *juliaPath = "juliares-" + std::string(argv[i + 1]) + ".txt";
        }
    }
}


int main (int argc, char* argv[]) {
    double exec_time_schur, exec_time_arnoldi, exec_time;

    double t = 1;
    //input values
    double alpha = 0.03754079961218426;
    double beta = 1;

    string mtxPath;
    string juliaPath;
    int krylovDegree;
    double rtol;

    cerr << "mkl max threads: " << mkl_get_max_threads() << endl;
    cerr << "omp max threads: " << omp_get_max_threads() << endl;


    if(argc != 5){
        cerr << "Usage: " << argv[0] << " -k <krylov-degree> -m <mtxSize>" << endl;
        return 1;
    }

    processArgs(argc, argv, &krylovDegree, &mtxPath, &juliaPath);
    readJuliaVec(juliaPath);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMtx(mtxPath);
    int size = (int) A.getSize();

    dense_vector b = dense_vector(size);
    b.insertValue(0, 1);
    //b.insertValue(floor(size/2), 1);
    double betaVal = b.getNorm2();

    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    exec_time = -omp_get_wtime();
    exec_time_arnoldi = -omp_get_wtime();
    arnoldiIteration(A, b, krylovDegree, size, &V, &H);
    exec_time_arnoldi += omp_get_wtime();

    H = -H;

    exec_time_schur = -omp_get_wtime();
    dense_matrix mlfH = calculate_MLF((double *) H.getDataPointer(), alpha, beta, krylovDegree);

    exec_time_schur += omp_get_wtime();

    dense_vector res = getApproximation(V, mlfH, betaVal);

    exec_time += omp_get_wtime();

    dense_vector diff = res - juliares;

    double diffNorm = cblas_dnrm2(size, diff.values.data(), 1);
    double trueNorm = cblas_dnrm2(size, juliares.values.data(), 1);

    cout << exec_time_schur << "," << (double) diffNorm / trueNorm << endl;
    
    mkl_sparse_destroy(A.getMKLSparseMatrix());


    return 0;
}