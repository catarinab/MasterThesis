#ifndef MLF_HPP
#define MLF_HPP

#include <complex>

#include "dense_matrix.hpp"

pair<double *, vector<vector<int>>> calculate_MLF(double * A, double alpha, double beta, int size);

#endif