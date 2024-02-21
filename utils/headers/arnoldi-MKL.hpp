#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"


int arnoldiIteration(csr_matrix A, dense_vector b, int k_total, int m, dense_matrix * V, dense_matrix * H);