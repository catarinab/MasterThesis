#include <complex>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

#include "headers/mittag_leffler_scalar.hpp"
#include "mkl.h"
#include "headers/dense_matrix.hpp"

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


dense_matrix calculate_MLF(double * T, double alphaInput, double betaInput, int size) {
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

    dense_matrix resultMatrix = dense_matrix(size, size);

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            resultMatrix.setValue(i, j, result(i,j).real());
        }
    }

    return resultMatrix;
}

