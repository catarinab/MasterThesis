#ifndef ARNOLDI_MKL_HPP
#define ARNOLDI_MKL_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"


int arnoldiIteration(const csr_matrix& A, dense_vector b, int k_total, int m, dense_matrix * V, dense_matrix * H);

#endif // ARNOLDI_MKL_HPP