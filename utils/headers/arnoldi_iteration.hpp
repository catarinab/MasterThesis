#ifndef ARNOLDIITERATION_HPP
#define ARNOLDIITERATION_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"

int restartedArnoldiIteration(const csr_matrix& A, dense_vector& b, int k_total, int m, int me, dense_matrix * V,
                              dense_matrix * H);

int arnoldiIteration(const csr_matrix& A, dense_vector& b, int k_total, int m, int me, dense_matrix * V,
                     dense_matrix * H);

#endif // ARNOLDIITERATION_HPP