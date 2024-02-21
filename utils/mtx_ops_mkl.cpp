#include <iostream>
#include <vector>
#include <omp.h>
#include <mkl.h>

#include <Eigen/Dense>

using namespace Eigen;

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

//multiply dense matrix and dense vector
dense_vector  denseMatrixVec(dense_matrix  A, dense_vector  b) {
    dense_vector m(A.getRowVal());
    cblas_dgemv(CblasColMajor, CblasNoTrans, A.getRowVal(), A.getColVal(), 1.0, 
    A.getDataPointer(), A.getRowVal(), b.values.data(), 1, 0.0, m.values.data(), 1);
    return m;
}

//multiply two dense matrices
dense_matrix  denseMatrixMult(dense_matrix A, dense_matrix B) {
    dense_matrix C(A.getRowVal(), B.getColVal());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            A.getRowVal(), B.getColVal(), A.getColVal(),
            1.0, A.getDataPointer(), A.getRowVal(),
            B.getDataPointer(), B.getColVal(),
            0.0, C.getDataPointer(), C.getRowVal());
    return C;
}

dense_vector  vectorElementWiseDivision(dense_vector  a, dense_vector  b) {
    dense_vector res(a.getSize());
    for (int i = 0; i < a.getSize(); i++) {
        res.insertValue(i, a.getValue(i) / b.getValue(i));
    }
    return res;
}

dense_matrix  denseMatrixAdd(dense_matrix A, dense_matrix b) {
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

//convert dense_matrix to Eigen MatrixXd
MatrixXd convertDenseEigenMtx(dense_matrix A) {
    MatrixXd eigenMtx(A.getRowVal(), A.getColVal());

    return Map<MatrixXd>(A.getDataPointer(), A.getRowVal(), A.getColVal());
}

//convert Eigen MatrixXd to dense_matrix
dense_matrix convertEigenDenseMtx(MatrixXd A) {
    dense_matrix denseMtx(A.rows(), A.cols());
    
    denseMtx.setValues(vector<double>(A.data(), A.data() + A.size()));
    return denseMtx;

}

//solve linear system using Eigen library and LU decomposition
dense_matrix  solveEq(dense_matrix A, dense_matrix b) {
    MatrixXd eigenMtxA = convertDenseEigenMtx(A);
    MatrixXd eigenMtxB = convertDenseEigenMtx(b);

    MatrixXd res = eigenMtxA.partialPivLu().solve(eigenMtxB);

    return convertEigenDenseMtx(res);
}

//multiply sparse matrix and dense vector
dense_vector  sparseMatrixVector(csr_matrix matrix, dense_vector  vec) {
    dense_vector  res(vec.getSize());

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.getMKLSparseMatrix(), matrix.getMKLDescription(),
                    vec.values.data(), 0.0, res.values.data());

    return res;
}

//b = a*scalar + b
dense_vector addVec(dense_vector b, dense_vector a, double scalar) {
    cblas_daxpy(a.size, scalar, a.values.data(), 1, b.values.data(), 1);
    return b;
}

//dot product of two dense vectors
double dotProduct(dense_vector a, dense_vector b) {
    return cblas_ddot(a.size, a.values.data(), 1, b.values.data(), 1);
}

double vectorTwoNorm(dense_vector  vec) {
    return cblas_dnrm2(vec.getSize(), vec.values.data(), 1.0);
}
