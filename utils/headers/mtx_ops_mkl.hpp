#include "dense_vector.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"

using namespace std;

//build sparse matrix from matrix market file
csr_matrix buildFullMtx(const string& input_file);

csr_matrix buildPartMatrix(const string& input_file, int me, int * displs, int * counts) ;


void checkValues(int a, int b, const string& func);

//multiply dense matrix and dense vector
dense_vector denseMatrixVec(dense_matrix A, dense_vector b);

//multiply two dense matrices
dense_matrix denseMatrixMult(dense_matrix A, dense_matrix B);

dense_matrix denseMatrixAdd(dense_matrix A, dense_matrix b);

//Subtract two dense matrices
dense_matrix denseMatrixSub(dense_matrix A, dense_matrix b);

//solve linear system using Eigen library and LU decomposition
dense_matrix solveEq(dense_matrix A, dense_matrix b);

//multiply sparse matrix and dense vector
dense_vector sparseMatrixVector(csr_matrix matrix, dense_vector vec);

//add two dense vectors
dense_vector addVec(dense_vector a, dense_vector b);

//dot product of two dense vectors
double dotProduct(dense_vector a, dense_vector b);

double vectorTwoNorm(dense_vector vec);
