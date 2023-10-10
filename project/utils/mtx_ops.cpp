#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>
#include "csr_matrix.cpp"
#include "Vector.cpp"
#include "io_ops.cpp"

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