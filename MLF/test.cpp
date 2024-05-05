#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <fstream>
#include <iomanip>

#include "../utils/headers/dense_matrix.hpp"
#include "../utils/headers/calculate-MLF.hpp"

using namespace std;

int main (int argc, char* argv[]) {

    double alpha = 0.5;
    double beta = 0;

    if(argc != 2){
        cerr << "Usage: " << argv[0] << " <size>" << endl;
        return 1;
    }

    int size = stoi(argv[1]);

    string folder = string("krylov-784/"+to_string(size)+"/");

    ofstream outputFile;

    outputFile.open(folder + "/cpp.txt");

    dense_matrix A;

    A.readVector("krylov-784/"+to_string(size)+"-vector.txt");

    double exec_time = -omp_get_wtime();

    dense_matrix B = calculate_MLF((double *) A.getDataPointer(), alpha, beta, size);

    exec_time += omp_get_wtime();

    cout << exec_time << endl;

    return 0;
}