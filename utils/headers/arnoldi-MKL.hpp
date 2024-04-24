#ifndef ARNOLDI_MKL_HPP
#define ARNOLDI_MKL_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"


int arnoldiIteration(const csr_matrix& A, const dense_vector& initVec, int k_total, int m, dense_matrix * V,
                     dense_matrix * H, int nu);

#endif // ARNOLDI_MKL_HPP