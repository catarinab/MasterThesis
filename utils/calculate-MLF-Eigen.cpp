#include <complex>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

#include "headers/Evaluate-Single-ML.hpp"
#include "mkl.h"

using namespace std;

/*
Algorithm based on the paper "Computing the matrix Mittag–Leffler function with applications to fractional calculus"
by Roberto Garrappa and Marina Popolizio
*/

double alpha;
double betaVal;

complex<double> evaluateSingleWrapper(complex<double> z, int k) {
    return evaluateSingle(z, alpha, betaVal, k);
}


complex<double> * calculate_MLF(complex<double> * T, double alphaInput, double betaInput, int size) {
    alpha = alphaInput;
    betaVal = betaInput;

    cout << "calculating with eigen" << endl;

    Eigen::MatrixXcd A(size, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            A(i,j) = T[i*size + j];
        }
    }

    Eigen::MatrixXcd result = A.matrixFunction(evaluateSingleWrapper);

    auto * resultArray = (complex<double> *) malloc(size * size * sizeof(complex<double>));

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            resultArray[i*size + j] = {result(i,j).real(), result(i,j).imag()};
        }
    }

    return resultArray;
}

