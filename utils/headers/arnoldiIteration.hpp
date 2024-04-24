#ifndef ARNOLDIITERATION_HPP
#define ARNOLDIITERATION_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"


int arnoldiIteration(const csr_matrix& A, const dense_vector& b, int k_total, int m, int me, int nprocs, dense_matrix * V,
                     dense_matrix * H, int nu);

#endif // ARNOLDIITERATION_HPP