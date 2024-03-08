#ifndef SCHUR_HPP
#define SCHUR_HPP

#include "dense_matrix.hpp"
#include "utils.hpp"

vector<vector<int>> schurDecomposition(lapack_complex_double ** lapacke_A, lapack_complex_double ** U, int size);

#endif