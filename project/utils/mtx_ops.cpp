#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>
#include "csr_matrix.cpp"
#include "io_ops.cpp"
#ifndef VEC
#define VEC 1
    #include "Vector.cpp"
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

dense_Matrix denseMatrixMatrixMult(dense_Matrix A, dense_Matrix b) {
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

dense_Matrix denseMatrixMatrixAdd(dense_Matrix A, dense_Matrix b) {
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

dense_Matrix denseMatrixMatrixSub(dense_Matrix A, dense_Matrix b) {
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

void getCofactor(dense_Matrix A, dense_Matrix * interRes ,int p, int q){
    int rowIndex = 0, colIndex = 0;
    int n = A.getRowVal();

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row != p && col != q) {
                interRes->setValue(rowIndex, colIndex++, A.getValue(row, col));
                
                if (colIndex == n - 1) {
                    colIndex = 0;
                    rowIndex++;
                }
            }
        }
    }
}

double getDeterminant(dense_Matrix A) {
    int n = A.getRowVal();
    double det = 0;

    if (n == 1)
        return A.getValue(0, 0);

    dense_Matrix interRes(n, n);

    int sign = 1;

    for (int f = 0; f < n; f++) {
        getCofactor(A, &interRes, 0, f);
        det += sign * A.getValue(0, f) * getDeterminant(interRes);
        sign = -sign;
    }

    return det;
}

dense_Matrix denseMatrixInverse(dense_Matrix A) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    if(rows != cols) {
        cout << "Error getting matrix inverse -> matrix not square: " << endl;
        cout << "A: " << A.getRowVal() << "x" << A.getColVal() << endl;
        exit(1);
    }

    double det = getDeterminant(A);
}

Vector sparseMatrixVector(CSR_Matrix matrix, Vector vec, int begin, int end, int size) {
    Vector res(end - begin);
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


Vector subtractVec(Vector a, Vector b, int begin, int end) {
    Vector res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] - b.values[i]);
    }
    return res;
}

Vector addVec(Vector a, Vector b, int begin, int end) {
    Vector res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] + b.values[i]);
    }
    return res;
}

double dotProduct(Vector a, Vector b, int begin, int end) {
	double dotProd = 0.0;

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a.values[i] * b.values[i]);
	}
    return dotProd;
}