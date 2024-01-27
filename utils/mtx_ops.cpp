#include <iostream>
#include <vector>
#include <omp.h>

#include <Eigen/Dense>
using namespace Eigen;

#include "headers/dense_vector.hpp"
#include "headers/dense_matrix.hpp"
#include "headers/io_ops.hpp"
#include "headers/csr_matrix.hpp"

using namespace std;

//build sparse matrix from matrix martket file
csr_matrix buildFullMtx(string input_file) {
    int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_full_mtx(input_file, &rows, &cols, &nz);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    return csr;
} 

csr_matrix buildPartMatrix(string input_file, int me, int * displs, int * counts) {
    int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_part_mtx(input_file, &rows, &cols, &nz, displs, counts, me);
    csr_matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    return csr;
} 



void checkValues(int a, int b, string func) {
    if(a != b) {
        cout << "Error: " << func << ": a != b" << endl;
        cout << "a: " << a << endl;
        cout << "b: " << b << endl;
        exit(1);
    }

}

//multiply dense matrix and dense vector
dense_vector denseMatrixVec(dense_matrix A, dense_vector b) {
    int rows = A.getRowVal();
    int cols = A.getColVal();

    checkValues(cols, b.getSize(), "denseMatrixVec");

    dense_vector res(rows);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double resVal = 0;
        for (int j = 0; j < cols; j++) {
            resVal += A.getValue(i, j) * b.values[j];
        }
        res.insertValue(i, resVal);
    }
    return res;
}

//multiply two dense matrices
dense_matrix denseMatrixMult(dense_matrix A, dense_matrix b) {
    int endCols = b.getColVal();
    int endRows = A.getRowVal();

    checkValues(A.getColVal(), b.getRowVal(), "denseMatrixMult");
    
    dense_matrix res(endRows, endCols);

    #pragma omp parallel for
    for (int i = 0; i < endRows; i++) {
        for (int j = 0; j < endCols; j++) {
            double resVal = 0;
            for (int k = 0; k < A.getColVal(); k++) {
                resVal += A.getValue(i, k) * b.getValue(k, j);
            }
            res.setValue(i, j, resVal);
        }
    }
    return res;
}

//Add two dense matrices
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

//convert dense_matrix to Eigen MatrixXd
MatrixXd convertDenseEigenMtx(dense_matrix A) {
    MatrixXd eigenMtx(A.getRowVal(), A.getColVal());

    #pragma omp parallel for
    for(int i = 0; i < A.getRowVal(); i++) {
        for(int j = 0; j < A.getColVal(); j++) {
            eigenMtx(i, j) = A.getValue(i, j);
        }
    }
    return eigenMtx;
}

//convert Eigen MatrixXd to dense_matrix
dense_matrix convertEigenDenseMtx(MatrixXd A) {
    dense_matrix denseMtx(A.rows(), A.cols());
    
    #pragma omp parallel for
    for(int i = 0; i < A.rows(); i++) {
        for(int j = 0; j < A.cols(); j++) {
            denseMtx.setValue(i, j, A(i, j));
        }
    }
    return denseMtx;

}

//solve linear system using Eigen library and LU decomposition
dense_matrix solveEq(dense_matrix A, dense_matrix b) {
    MatrixXd eigenMtxA = convertDenseEigenMtx(A);
    MatrixXd eigenMtxB = convertDenseEigenMtx(b);

    MatrixXd res = eigenMtxA.partialPivLu().solve(eigenMtxB);

    return convertEigenDenseMtx(res);
}

//multiply sparse matrix and dense vector
dense_vector sparseMatrixVector(csr_matrix matrix, dense_vector vec, int begin, int end) {
    dense_vector res(end - begin);
    int resIndex = 0;
    double resVal = 0;

    checkValues(matrix.getSize(), vec.getSize(), "sparseMatrixVector");

    if(matrix.getNZ() == 0 || vec.values.size() == 0) 
        return res;


    #pragma omp parallel for private(resIndex, resVal) schedule(guided)
    for(int i = begin; i < end; i++) {
        resIndex = i - begin;
        resVal = 0;


        vector<SparseTriplet> row = matrix.getRow(i);
        if(row.size() == 0) continue;

        for(int j = 0; j < row.size(); j++) {
            int col = row[j].col;
            resVal += row[j].value * vec.values[col];
        }
        
        res.insertValue(resIndex, resVal);
    }
    return res;
}

//subtract two dense vectors
dense_vector subtractVec(dense_vector a, dense_vector b, int begin, int end) {
    dense_vector res(end - begin);
    int resIndex = 0;

    checkValues(a.getSize(), b.getSize(), "subtractVec");

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] - b.values[i]);
    }
    return res;
}

//add two dense vectors
dense_vector addVec(dense_vector a, dense_vector b, int begin, int end) {
    dense_vector res(end - begin);
    int resIndex = 0;

    checkValues(a.getSize(), b.getSize(), "addVec");

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res.insertValue(resIndex, a.values[i] + b.values[i]);
    }
    return res;
}

//dot product of two dense vectors
double dotProduct(dense_vector a, dense_vector b, int begin, int end) {
	double dotProd = 0.0;

    checkValues(a.getSize(), b.getSize(), "dotProduct");

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a.values[i] * b.values[i]);
	}
    return dotProd;
}
