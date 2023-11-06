#include <iostream>
#include <vector>
#include <omp.h>
#include "csr_matrix.cpp"
#include "io_ops.cpp"

#include <Eigen/Dense>
using namespace Eigen;

#ifndef VEC
#define VEC 1
    #include "DenseVector.cpp"
#endif
#ifndef DM
#define DM 1
    #include "../utils/dense_Matrix.cpp"
#endif

using namespace std;

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
CSR_Matrix buildMtx(string input_file) {
    int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_mtx(input_file, &rows, &cols, &nz);
    CSR_Matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    return csr;
}

dense_Matrix denseMatrixVec(dense_Matrix A, DenseVector b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(cols != b.getSize()) {
        cout << "Error: A.getColVal() != b.getSize()" << endl;
        cout << "A: " << A.getRowVal() << "x" << A.getColVal() << endl;
        cout << "b: " << b.getSize() << endl;
        exit(1);
    }

    dense_Matrix res(rows, 1);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double resVal = 0;
        #pragma omp parallel for reduction(+:resVal)
        for (int j = 0; j < cols; j++) {
            resVal += A.getValue(i, j) * b.values[j];
        }
        res.setValue(i, 0, resVal);
    }
    return res;
}

dense_Matrix denseMatrixMult(dense_Matrix A, dense_Matrix b) {
    int endCols = b.getColVal();
    int endRows = A.getRowVal();
    double resVal = 0;

    if(A.getColVal() != b.getRowVal()) {
        cout << "Error: MULT: A.getColVal() != b.getRowVal()" << endl;
        cout << "A: " << A.getRowVal() << "x" << A.getColVal() << endl;
        cout << "b: " << b.getRowVal() << "x" << b.getColVal() << endl;
        exit(1);
    }
    
    dense_Matrix res(endRows, endCols);

    for (int i = 0; i < endRows; i++) {
        for (int j = 0; j < endCols; j++) {
            resVal = 0;
            #pragma omp parallel for reduction(+:resVal)
            for (int k = 0; k < A.getColVal(); k++) {
                resVal += A.getValue(i, k) * b.getValue(k, j);
            }
            res.setValue(i, j, resVal);
        }
    }
    return res;
}

dense_Matrix denseMatrixAdd(dense_Matrix A, dense_Matrix b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(cols != b.getColVal() || rows != b.getRowVal()) {
        cout << "Error: ADD: A.getColVal() != b.getColVal() || A.getRowVal() != b.getRowVal()" << endl;
        cout << "A: " << A.getRowVal() << "x" << A.getColVal() << endl;
        cout << "b: " << b.getRowVal() << "x" << b.getColVal() << endl;
        exit(1);
    }

    dense_Matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) + b.getValue(i, j));
        }
    }
    return res;
}

dense_Matrix denseMatrixSub(dense_Matrix A, dense_Matrix b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(cols != b.getColVal() || rows != b.getRowVal()) {
        cout << "Error: ADD: A.getColVal() != b.getColVal() || A.getRowVal() != b.getRowVal()" << endl;
        cout << "A: " << A.getRowVal() << "x" << A.getColVal() << endl;
        cout << "b: " << b.getRowVal() << "x" << b.getColVal() << endl;
        exit(1);
    }

    dense_Matrix res(rows, cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.setValue(i, j, A.getValue(i, j) - b.getValue(i, j));
        }
    }
    return res;
}

MatrixXd convertDenseEigenMtx(dense_Matrix A) {
    MatrixXd eigenMtx(A.getRowVal(), A.getColVal());
    for(int i = 0; i < A.getRowVal(); i++) {
        #pragma omp parallel for
        for(int j = 0; j < A.getColVal(); j++) {
            eigenMtx(i, j) = A.getValue(i, j);
        }
    }
    return eigenMtx;
}

dense_Matrix convertEigenDenseMtx(MatrixXd A) {
    dense_Matrix denseMtx(A.rows(), A.cols());
    for(int i = 0; i < A.rows(); i++) {
        #pragma omp parallel for
        for(int j = 0; j < A.cols(); j++) {
            denseMtx.setValue(i, j, A(i, j));
        }
    }
    return denseMtx;

}

dense_Matrix solveEq(dense_Matrix A, dense_Matrix b) {
    MatrixXd eigenMtxA = convertDenseEigenMtx(A);
    MatrixXd eigenMtxB = convertDenseEigenMtx(b);

    MatrixXd res = eigenMtxA.partialPivLu().solve(eigenMtxB);

    return convertEigenDenseMtx(res);
}

DenseVector sparseMatrixVector(CSR_Matrix matrix, DenseVector vec, int begin, int end, int size) {
    DenseVector res(end - begin);
    int resIndex = 0;
    int mtxPtr = 0;
    double resVal = 0;

    if(matrix.getNZ() == 0 || vec.values.size() == 0) 
        return res;

    #pragma omp parallel for private(resIndex, mtxPtr, resVal)
    for(int i = begin; i < end; i++) {
        resIndex = i - begin;
        mtxPtr = 0;
        resVal = 0;

        vector<SparseTriplet> row = matrix.getRow(i);
        if(row.size() == 0) continue;

        for(int j = 0; j < size; j++) {
            if(row[mtxPtr].col == j) {
                resVal += row[mtxPtr].value * vec.values[j];
                mtxPtr++;
            }
            //os restantes elementos de um deles sao zero
            if(mtxPtr == row.size()) break;
        }
        if(resVal != 0)
            res.insertValue(resIndex, resVal);
    }

    return res;
}

DenseVector subtractVec(DenseVector a, DenseVector b, int begin, int end) {
    DenseVector res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] - b.values[i]);
    }
    return res;
}

DenseVector addVec(DenseVector a, DenseVector b, int begin, int end) {
    DenseVector res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] + b.values[i]);
    }
    return res;
}

double dotProduct(DenseVector a, DenseVector b, int begin, int end) {
	double dotProd = 0.0;

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a.values[i] * b.values[i]);
	}
    return dotProd;
}