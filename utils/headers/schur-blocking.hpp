#ifndef SCHUR_HPP
#define SCHUR_HPP

#include "dense_matrix.hpp"
#include "utils.hpp"
#include <complex>


std::vector<std::vector<int>> schurDecomposition(double * A, std::complex<double> ** T, std::complex<double> ** U, int size);

#endif