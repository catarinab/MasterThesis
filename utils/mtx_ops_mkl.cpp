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
csr_matrix buildFullMtx(const string& input_file) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_full_mtx(input_file, &rows, &cols, &nz);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    csr.defineMKLSparseMatrix();
    return csr;
}

csr_matrix buildPartMatrix(const string& input_file, int me, int * displs, int * counts) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_part_mtx(input_file, &rows, &cols, &nz, displs, counts, me);
    csr_matrix csr(counts[me]);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    csr.defineMKLSparseMatrix();
    return csr;
}


void checkValues(int a, int b, const string& func) {
    if(a != b) {
        cout << "Error: " << func << ": a != b" << endl;
        cout << "a: " << a << endl;
        cout << "b: " << b << endl;
        exit(1);
    }

}

dense_matrix solveEq(const dense_matrix& A, dense_matrix b) {
    lapack_int ipiv[A.getRowVal()];
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.getRowVal(), A.getColVal(), (double *) A.getDataPointer(),
                   A.getRowVal(), ipiv);
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', A.getRowVal(), b.getColVal(), (double *) A.getDataPointer(),
                   A.getRowVal(), ipiv, (double *) b.getDataPointer(), b.getColVal());

    return b;
}



//multiply dense matrix and dense vector
dense_vector denseMatrixVec(const dense_matrix& A, const dense_vector& b) {
    dense_vector m(A.getRowVal());
    cblas_dgemv(CblasColMajor, CblasNoTrans, A.getRowVal(), A.getColVal(), 1.0,
                A.getDataPointer(), A.getRowVal(), b.values.data(), 1, 0.0, m.values.data(), 1);
    return m;
}

//multiply two dense matrices
dense_matrix denseMatrixMult(const dense_matrix& A, const dense_matrix& B) {
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

    checkValues(cols, B.getColVal(), "denseMatrixAdd");
    checkValues(rows, B.getRowVal(), "denseMatrixAdd");

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

    checkValues(cols, B.getColVal(), "denseMatrixSub");
    checkValues(rows, B.getRowVal(), "denseMatrixSub");

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
dense_vector sparseMatrixVector(const csr_matrix& matrix, const dense_vector& vec) {
    dense_vector res(vec.getSize());

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.getMKLSparseMatrix(), matrix.getMKLDescription(),
                    vec.values.data(), 0.0, res.values.data());

    return res;
}

//a = a + scalar*x
//this function could be used to add or subtract vectors (only the vector x is multiplied by the scalar)
dense_vector addVec(dense_vector a, const dense_vector& b, double scalar, int size) {
    cblas_daxpy(size, scalar, b.values.data(), 1, a.values.data(), 1);
    return a;
}

//dot product of two dense vectors
double dotProduct(const dense_vector& a, const dense_vector& b, int size) {
    return cblas_ddot(size, a.values.data(), 1, b.values.data(), 1);
}

double vectorTwoNorm(const dense_vector& vec) {
    return cblas_dnrm2(vec.getSize(), vec.values.data(), 1.0);
}
