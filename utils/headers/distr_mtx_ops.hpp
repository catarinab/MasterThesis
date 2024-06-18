#ifndef DISTR_MTX_OPS_HPP
#define DISTR_MTX_OPS_HPP

#include "dense_vector.hpp"
#include "csr_matrix.hpp"

//global variables for scattering and gathering
extern int *displs;
extern int *counts;
extern int helpSize;

void initGatherVars(int size, int nprocs);
void initGatherVarsFullMtx(int size, int nprocs);

double distrDotProduct(dense_vector& a, dense_vector& b, int size, int me);

void distrSumOp(dense_vector& a, dense_vector& b, double scalar, int size, int me);

void distrMatrixVec(const csr_matrix& A, dense_vector& vec, dense_vector& res, int size);

double distrNorm(dense_vector& a, int size, int me);

#endif // DISTR_MTX_OPS_HPP