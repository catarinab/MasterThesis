#ifndef PADE_EXP_APPROX_UTILS_HPP
#define PADE_EXP_APPROX_UTILS_HPP

#include <vector>
#include "dense_matrix.hpp"

using namespace std;

int findM(double norm, int theta , int * power);


vector<double> get_pade_coefficients(int m);

dense_matrix definePadeParams(vector<dense_matrix> * powers, int * m, int * power, dense_matrix A);

#endif // PADE_EXP_APPROX_UTILS_HPP