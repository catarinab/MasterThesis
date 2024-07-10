#ifndef MTX_OPS_MKL_HPP
#define MTX_OPS_MKL_HPP


#include "dense_vector.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include "utils.hpp"
#include <mkl.h>

using namespace std;

//build sparse matrix from matrix market file
csr_matrix buildFullMatrix(const string& input_file);

csr_matrix buildPartialMatrix(const string& input_file, int me, int * displs, int * counts);

csr_matrix buildPartIndentityMatrix(int me, int * displs, int * counts);

void checkValues(int a, int b, const string& func);

dense_matrix solveEq(const dense_matrix& A, const dense_matrix& b);

//multiply dense matrix and dense vector
dense_vector denseMatrixVec(const dense_matrix& A, const dense_vector& b);

//multiply two dense matrices
dense_matrix denseMatrixMult(const dense_matrix& A, const dense_matrix& B) ;

dense_matrix denseMatrixAdd(const dense_matrix& A, const dense_matrix& B);

//Subtract two dense matrices
dense_matrix denseMatrixSub(const dense_matrix& A, const dense_matrix& B);

//multiply sparse matrix and dense vector
void sparseMatrixVector(const csr_matrix& matrix, const dense_vector& vec, dense_vector& res);

//add two dense vectors
void addVec(dense_vector& y, const dense_vector& x, double scalar, int size);

//dot product of two dense vectors
double dotProduct(const dense_vector& a, const dense_vector& b, int size);

#endif