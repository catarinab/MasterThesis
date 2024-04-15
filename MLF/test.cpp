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

    double matlabInf;
    double matlabFro;
    double condNumber;

    if(argc != 2){
        cerr << "Usage: " << argv[0] << " <size>" << endl;
        return 1;
    }

    int size = stoi(argv[1]);

    string folder = string("krylov/"+to_string(size)+"/");

    ofstream outputFile;

    outputFile.open(folder + "/cpp.txt");

    dense_matrix A;

    A.readVector("krylov/"+to_string(size)+"-vector.txt");

    outputFile << "Matrix with size " << size << endl;

    ifstream cond(folder + "cond.txt");

    cond >> condNumber;

    cond.close();

    ifstream inf(folder + "infNorm.txt");

    inf >> matlabInf;

    inf.close();

    ifstream fro(folder + "fro.txt");

    fro >> matlabFro;

    fro.close();

    double exec_time;

    exec_time = -omp_get_wtime();

    pair <double *, vector<vector<int>>> res = calculate_MLF(A.getDataPointer(), alpha, beta, size);

    double * B = res.first;
    vector<vector<int>> ind = res.second;

    for(int i = 2; i < size; i++){
        int count = 0;
        for(const auto & j : ind){
            if(j.size() == i)
                count++;
        }
        if(count > 0)
            outputFile << count << " blocks of size " << i << ", ";
    }
    outputFile << endl;

    exec_time += omp_get_wtime();

    double infNorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'I', size, size, B, size);

    cout << "Inf Norm: " << infNorm << setprecision(16) << endl;

    double froNorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'F', size, size, B, size);

    cout << "Fro Norm: "<< setprecision(16)  << froNorm << endl;

    outputFile << "numero de condicionamento: " << condNumber << endl;
    outputFile << "norma Infinita c++: " << scientific << infNorm << endl;
    outputFile << "norma Frobenius c++: " << scientific << froNorm << endl;

    cerr << "Matlab Inf: " << matlabInf << endl;

    cerr << "Matlab Fro: " << matlabFro << endl;

    outputFile << "erro frobenius: " << scientific << (abs(infNorm - matlabInf) / matlabInf) * 100 << "%" << endl;

    outputFile << "erro inf: " << scientific << (abs(froNorm - matlabFro) / matlabFro) * 100 << "%" << endl << endl;

    cout << "Execution time: " << exec_time << endl;

    outputFile.close();

    free(B);

    return 0;
}