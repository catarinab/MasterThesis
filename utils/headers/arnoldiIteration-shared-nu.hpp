#ifndef ARNOLDI_ITERATION_SHARED_NU_HPP
#define ARNOLDI_ITERATION_SHARED_NU_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"

int arnoldiIteration(const csr_matrix &A, dense_vector &initVec, int k_total, int m, dense_matrix *V,
                     dense_matrix *H, int nu);

#endif // ARNOLDI_ITERATION_SHARED_NU_HPP