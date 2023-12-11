#ifndef MTX_OPS_HPP
#define MTX_OPS_HPP

#include <string>
#include "dense_vector.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
csr_matrix buildMtx(string input_file);

void checkValues(int a, int b, string func);

dense_vector denseMatrixVec(dense_matrix A, dense_vector b);

dense_matrix denseMatrixMult(dense_matrix A, dense_matrix b);

dense_matrix denseMatrixAdd(dense_matrix A, dense_matrix b);

dense_matrix denseMatrixSub(dense_matrix A, dense_matrix b);

MatrixXd convertDenseEigenMtx(dense_matrix A);

dense_matrix convertEigenDenseMtx(MatrixXd A);


dense_matrix solveEq(dense_matrix A, dense_matrix b) ;

dense_vector sparseMatrixVector(csr_matrix matrix, dense_vector vec, int begin, int end, int size) ;

dense_vector subtractVec(dense_vector a, dense_vector b, int begin, int end);

dense_vector addVec(dense_vector a, dense_vector b, int begin, int end);

double dotProduct(dense_vector a, dense_vector b, int begin, int end);

#endif // MTX_OPS_HPP