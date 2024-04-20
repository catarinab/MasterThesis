#ifndef DISTR_MTX_OPS_HPP
#define DISTR_MTX_OPS_HPP

#include "dense_vector.hpp"
#include "csr_matrix.hpp"

//global variables for scattering and gathering
extern int *displs;
extern int *counts;
extern int helpSize;

void initGatherVars(int size, int nprocs);

void sendVectors(dense_vector a, dense_vector b, int helpSize, int func, int size);

double distrDotProduct(dense_vector a, const dense_vector& b, int size, int me, int nprocs);

dense_vector distrSumOp(dense_vector a, const dense_vector& b, double scalar, int size, int me, int nprocs);

dense_vector distrMatrixVec(csr_matrix A, const dense_vector& vec, int size, int me, int nprocs);

#endif // DISTR_MTX_OPS_HPP