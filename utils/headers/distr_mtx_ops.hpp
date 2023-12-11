#ifndef DISTR_MTX_OPS_HPP
#define DISTR_MTX_OPS_HPP

#include "dense_vector.hpp"
#include "csr_matrix.hpp"

//global variables
extern int *displs;
extern int *counts;
extern int helpSize;

void initGatherVars(int size, int nprocs);

void sendVectors(dense_vector a, dense_vector b, int helpSize, int func, int size);


double distrDotProduct(dense_vector a, dense_vector b, int size, int me, int nprocs);

dense_vector distrSubOp(dense_vector a, dense_vector b, int size, int me, int nprocs);

dense_vector distrSumOp(dense_vector a, dense_vector b, int size, int me, int nprocs);

dense_vector distrMatrixVec(csr_matrix A, dense_vector vec, int size, int me, int nprocs);

#endif // DISTR_MTX_OPS_HPP