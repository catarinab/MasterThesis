#include <iostream>
#include <omp.h>
#include <cstring>
#include <fstream>

#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/calculate_MLF.hpp"
#include "../utils/headers/arnoldi_iteration_shared.hpp"

using namespace std;

dense_vector matlabRes;

//Calculate the approximation of MLF(A)*b
dense_vector getApproximation(dense_matrix V, const dense_matrix& mlfH, double betaVal) {

    if(betaVal != 1)
        V = V * betaVal;

    return denseMatrixMult(V, mlfH).getCol(0);
}

void readVec(const string& filename) {
    ifstream inputFile(filename);
    if (inputFile) {
        double value;
        while (inputFile >> value) {
            matlabRes.values.push_back(value);
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
        else if(strcmp(argv[i], "-p") == 0) {
            *mtxPath = std::string(argv[i + 1]) + ".mtx";
            *juliaPath = std::string(argv[i + 1]) + "-res.txt";
        }
    }
}


int main (int argc, char* argv[]) {
    double exec_time_schur, exec_time_arnoldi, exec_time;
    //input values
    double alpha = 0.8;
    double beta = 0;

    string mtxPath;
    string vectorPath;
    int krylovDegree;

    cerr << "mkl max threads: " << mkl_get_max_threads() << endl;
    cerr << "omp max threads: " << omp_get_max_threads() << endl;


    if(argc != 5){
        cerr << "Usage: " << argv[0] << " -k <krylov-degree> -p <problemName>" << endl;
        return 1;
    }

    processArgs(argc, argv, &krylovDegree, &mtxPath, &vectorPath);

    cerr << "krylov degree: " << krylovDegree << endl;
    cerr << "mtxPath: " << mtxPath << endl;
    cerr << "vectorPath: " << vectorPath << endl;

    readVec(vectorPath);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMatrix(mtxPath);
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

    dense_vector diff = res - matlabRes;

    double diffNorm = cblas_dnrm2(size, diff.values.data(), 1);
    double trueNorm = cblas_dnrm2(size, matlabRes.values.data(), 1);

    cout << exec_time_schur << "," << std::scientific << (double) diffNorm / trueNorm << endl;
    
    mkl_sparse_destroy(A.getMKLSparseMatrix());

    return 0;
}