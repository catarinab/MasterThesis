#ifndef ARNOLDIITERATION_HPP
#define ARNOLDIITERATION_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"

double restartedArnoldiIteration_MLF(const csr_matrix& A, dense_vector& initVec, int k_total, int m, int me, dense_matrix * V,
                                 dense_matrix * H, double alpha, double beta);

int arnoldiIteration(const csr_matrix& A, dense_vector& b, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H);

#endif // ARNOLDIITERATION_HPP