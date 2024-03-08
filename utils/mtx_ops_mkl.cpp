#include <iostream>
#include <vector>
#include <omp.h>
#include <mkl.h>

#include "headers/dense_vector.hpp"
#include "headers/dense_matrix.hpp"
#include "headers/io_ops.hpp"
#include "headers/csr_matrix.hpp"

using namespace std;

//build sparse matrix from matrix market file
csr_matrix buildFullMtx(const string& input_file) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_full_mtx(input_file, &rows, &cols, &nz);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    return csr;
}

csr_matrix buildPartMatrix(const string& input_file, int me, int * displs, int * counts) {
    long long int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_part_mtx(input_file, &rows, &cols, &nz, displs, counts, me);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
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

lapack_complex_double lpck_z_sum(lapack_complex_double a, lapack_complex_double b) {
    lapack_complex_double result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

lapack_complex_double lpck_z_sub(lapack_complex_double a, lapack_complex_double b) {
    lapack_complex_double result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

lapack_complex_double lpck_z_mult(lapack_complex_double a, lapack_complex_double b) {
    lapack_complex_double result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

lapack_complex_double lpck_z_div(lapack_complex_double a, lapack_complex_double b) {
    lapack_complex_double result;
    double denominator = b.real * b.real + b.imag * b.imag;
    result.real = (a.real * b.real + a.imag * b.imag) / denominator;
    result.imag = (a.imag * b.real - a.real * b.imag) / denominator;
    return result;
}

ostream& operator << (ostream &os, const lapack_complex_double &a) {
    return (os << a.real << " + " << a.imag << "i ");
}

double lpck_abs(lapack_complex_double a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}


//multiply dense matrix and dense vector
dense_vector denseMatrixVec(dense_matrix  A, dense_vector  b) {
    dense_vector m(A.getRowVal());
    cblas_dgemv(CblasColMajor, CblasNoTrans, A.getRowVal(), A.getColVal(), 1.0, 
    A.getDataPointer(), A.getRowVal(), b.values.data(), 1, 0.0, m.values.data(), 1);
    return m;
}

//multiply two dense matrices
dense_matrix denseMatrixMult(dense_matrix A, dense_matrix B) {
    dense_matrix C(A.getRowVal(), B.getColVal());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            A.getRowVal(), B.getColVal(), A.getColVal(),
            1.0, A.getDataPointer(), A.getRowVal(),
            B.getDataPointer(), B.getColVal(),
            0.0, C.getDataPointer(), C.getRowVal());
    return C;
}

dense_matrix lapackeToDenseMatrix(lapack_complex_double * A, int rows, int cols) {
    dense_matrix res(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A[i * cols + j].real);
        }
    }
    return res;
}

dense_matrix denseMatrixAdd(dense_matrix A, dense_matrix b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    checkValues(cols, b.getColVal(), "denseMatrixAdd");
    checkValues(rows, b.getRowVal(), "denseMatrixAdd");

    dense_matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) + b.getValue(i, j));
        }
    }
    return res;
}

//Subtract two dense matrices
dense_matrix denseMatrixSub(dense_matrix A, dense_matrix b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    checkValues(cols, b.getColVal(), "denseMatrixSub");
    checkValues(rows, b.getRowVal(), "denseMatrixSub");

    dense_matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) - b.getValue(i, j));
        }
    }
    return res;
}

//multiply sparse matrix and dense vector
dense_vector sparseMatrixVector(csr_matrix matrix, dense_vector  vec) {
    dense_vector res(vec.getSize());

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.getMKLSparseMatrix(), matrix.getMKLDescription(),
                    vec.values.data(), 0.0, res.values.data());

    return res;
}

//y = y + scalar*x
dense_vector addVec(dense_vector y, dense_vector x, double scalar) {
    cblas_daxpy(x.size, scalar, x.values.data(), 1, y.values.data(), 1);
    return y;
}

//dot product of two dense vectors
double dotProduct(dense_vector a, dense_vector b) {
    return cblas_ddot(a.size, a.values.data(), 1, b.values.data(), 1);
}

double vectorTwoNorm(dense_vector  vec) {
    return cblas_dnrm2(vec.getSize(), vec.values.data(), 1.0);
}