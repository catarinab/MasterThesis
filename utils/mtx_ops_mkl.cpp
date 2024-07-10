#include <iostream>
#include <vector>
#include <mkl.h>

#include "headers/dense_vector.hpp"
#include "headers/dense_matrix.hpp"
#include "headers/io_ops.hpp"
#include "headers/csr_matrix.hpp"
#include "headers/mtx_ops_mkl.hpp"
#include "headers/utils.hpp"

using namespace std;

//build sparse matrix from matrix market file
csr_matrix buildFullMatrix(const string& input_file) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFileFullMtx(input_file, &rows, &cols, &nz);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    csr.defineMKLSparseMatrix();
    return csr;
}

csr_matrix buildPartialMatrix(const string& input_file, int me, int * displs, int * counts) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFilePartialMtx(input_file, &rows, &cols, &nz, displs, counts, me);
    csr_matrix csr(counts[me]);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    csr.defineMKLSparseMatrix();
    return csr;
}

csr_matrix buildPartIndentityMatrix(int me, int * displs, int * counts) {
    csr_matrix csr(counts[me]);
    for (int i = 0; i < counts[me]; i++) {
        csr.insertRow({SparseTriplet(i, i, 1)}, i);
    }
    csr.defineMKLSparseMatrix();
    return csr;
}

dense_matrix solveEq(const dense_matrix& A, const dense_matrix& b) {
    vector<lapack_int> ipiv(A.getRowVal());
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.getRowVal(), A.getColVal(), (double *) A.getDataPointer(),
                   A.getRowVal(), ipiv.data());
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', A.getRowVal(), b.getColVal(), (double *) A.getDataPointer(),
                   A.getRowVal(), ipiv.data(), (double *) b.getDataPointer(), b.getColVal());

    return b;
}

//multiply two dense matrices
dense_matrix denseMatrixMult(const dense_matrix& A, const dense_matrix& B) {
    if (A.getColVal() != B.getRowVal()) {
        throw std::invalid_argument("Matrices dimensions do not match for multiplication");
    }
    dense_matrix C(A.getRowVal(), B.getColVal());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                A.getRowVal(), B.getColVal(), A.getColVal(), 1.0,
                A.getDataPointer(), A.getRowVal(), B.getDataPointer(), B.getRowVal(), 0.0,
                (double *) C.getDataPointer(), C.getRowVal());

    return C;
}

dense_matrix denseMatrixAdd(const dense_matrix& A, const dense_matrix& B) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(rows != B.getRowVal() || cols != B.getColVal())
        throw std::invalid_argument("Matrices dimensions do not match for addition");

    dense_matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) + B.getValue(i, j));
        }
    }
    return res;
}

//Subtract two dense matrices
dense_matrix denseMatrixSub(const dense_matrix& A, const dense_matrix& B) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(rows != B.getRowVal() || cols != B.getColVal())
        throw std::invalid_argument("Matrices dimensions do not match for addition");

    dense_matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) - B.getValue(i, j));
        }
    }
    return res;
}

//multiply sparse matrix and dense vector
void sparseMatrixVector(const csr_matrix& matrix, const dense_vector& vec, dense_vector& res) {
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.getMKLSparseMatrix(), matrix.getMKLDescription(),
                    vec.values.data(), 0.0, res.values.data());
}

//a = a + scalar*x
//this function could be used to add or subtract vectors (only the vector x is multiplied by the scalar)
void addVec(dense_vector& a, const dense_vector& b, double scalar, int size) {
    cblas_daxpy(size, scalar, b.values.data(), 1, a.values.data(), 1);
}

//dot product of two dense vectors
double dotProduct(const dense_vector& a, const dense_vector& b, int size) {
    return cblas_ddot(size, a.values.data(), 1, b.values.data(), 1);
}

