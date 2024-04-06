#ifndef SCHUR_HPP
#define SCHUR_HPP

#include "dense_matrix.hpp"
#include "utils.hpp"

vector<vector<int>> schurDecomposition(double * A, complex<double> ** T, complex<double> ** U, int size);

#endif