#ifndef MTX_OPS_MKL_HPP
#define MTX_OPS_MKL_HPP


#include "dense_vector.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include <mkl.h>

using namespace std;

//build sparse matrix from matrix market file
csr_matrix buildFullMtx(const string& input_file);

csr_matrix buildPartMatrix(const string& input_file, int me, int * displs, int * counts) ;

void checkValues(int a, int b, const string& func);

lapack_complex_double lpck_z_sum(lapack_complex_double a, lapack_complex_double b);

lapack_complex_double lpck_z_sub(lapack_complex_double a, lapack_complex_double b);

lapack_complex_double lpck_z_mult(lapack_complex_double a, lapack_complex_double b);

lapack_complex_double lpck_z_div(lapack_complex_double a, lapack_complex_double b);

double lpck_abs(lapack_complex_double a);

ostream& operator << (ostream &os, const lapack_complex_double &a);

//multiply dense matrix and dense vector
dense_vector denseMatrixVec(dense_matrix A, dense_vector b);

//multiply two dense matrices
dense_matrix denseMatrixMult(dense_matrix A, dense_matrix B);

dense_matrix lapackeToDenseMatrix(lapack_complex_double * A, int rows, int cols);

dense_matrix  denseMatrixAdd(dense_matrix A, dense_matrix b);

//Subtract two dense matrices
dense_matrix  denseMatrixSub(dense_matrix A, dense_matrix b);

//solve linear system using Eigen library and LU decomposition
dense_matrix  solveEq(dense_matrix  A, dense_matrix  b);

//multiply sparse matrix and dense vector
dense_vector sparseMatrixVector(csr_matrix matrix, dense_vector vec);

//add two dense vectors
dense_vector addVec(dense_vector a, dense_vector b, double scalar);

//dot product of two dense vectors
double dotProduct(dense_vector a, dense_vector b);

double vectorTwoNorm(dense_vector vec);

#endif