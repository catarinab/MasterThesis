#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>
#include "csr_matrix.cpp"
#include "sparse_vec.cpp"
#include "io_ops.cpp"

using namespace std;

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix() {
    vector<vector<double>> A{
        {10, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {1, 20, 1, 2, 3, 4, 5, 6, 7, 8},
        {2, 1, 30, 1, 2, 3, 4, 5, 6, 7},
        {3, 2, 1, 40, 1, 2, 3, 4, 5, 6},
        {4, 3, 2, 1, 50, 1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1, 60, 1, 2, 3, 4},
        {6, 5, 4, 3, 2, 1, 70, 1, 2, 3},
        {7, 6, 5, 4, 3, 2, 1, 80, 1, 2},
        {8, 7, 6, 5, 4, 3, 2, 1, 90, 1},
        {9, 8, 7, 6, 5, 4, 3, 2, 1, 100}
    };
    return A;
}

CSR_Matrix buildMtx(string input_file) {
    int rows, cols, nz;
    vector<vector<SparseTriplet>> rowValues = readFile_mtx(input_file, &rows, &cols, &nz);
    CSR_Matrix csr(rows);
    for (int i = 0; i < rowValues.size(); i++) {
        csr.insertRow(rowValues[i], i);
    }
    return csr;
}

Sparse_Vec buildRandSparseVec(int size) {
    Sparse_Vec v(size, true);
    return v;
}

//construir tendo em conta as matrizes do matlab?
vector<double> buildVector() {
    vector<double> b{2, 8, 9, 2, 3, 4, 5, 6, 7, 8};
    return b;
}


vector<double> subtractVec(vector<double> a, vector<double> b, int begin, int end) {
    vector<double> res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res[resIndex] = a[i] - b[i];
    }
    return res;
}

Sparse_Vec subtractSparseVec(Sparse_Vec a, Sparse_Vec b, int begin, int end) {
    Sparse_Vec res(end - begin, false);
    int resIndex = 0;
    int vecPtrA = 0, vecPtrB = 0;

    if(a.nzValues.size() == 0) 
        return -b;
    
    else if(b.nzValues.size() == 0) 
        return a;
    

    for(int i = begin; i < end; i++) {
        if(a.nzValues[vecPtrA].col == i == b.nzValues[vecPtrB].col) {
            res.insertValue(SparseDouble(i, a.nzValues[vecPtrA].value - b.nzValues[vecPtrB].value));
            vecPtrA++; vecPtrB++;
        }
        else if(a.nzValues[vecPtrA].col == i){
            res.insertValue(SparseDouble(i, a.nzValues[vecPtrA].value));
            vecPtrA++;
        }
        else if(b.nzValues[vecPtrB].col == i){
            res.insertValue(SparseDouble(i, -b.nzValues[vecPtrB].value));
            vecPtrB++;
        }
        if(vecPtrA == a.nzValues.size() && vecPtrB == b.nzValues.size()) break;
    }
    return res;
}

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int begin, int end, int size) {
    vector <double> res(end - begin);
    int resIndex = 0;

	for (int i = begin; i < end; i++) {
        resIndex = i - begin;
		res[resIndex] = 0;
		for (int j = 0; j < size; j++) {
			res[resIndex] += matrix[i][j] * v[j];
		}
	}
	return res;
}

Sparse_Vec sparseMatrixVector(CSR_Matrix matrix, Sparse_Vec vec, int begin, int end, int size) {
    Sparse_Vec res(end - begin, false);

    if(matrix.getNZ() == 0 || vec.nzValues.size() == 0) 
        return res;

    #pragma omp parallel for
    for(int i = 0; i < end; i++) {
        int mtxPtr = 0, vecPtr = 0;
        vector<SparseTriplet> row = matrix.getRow(i);
        if(row.size() == 0) continue;

        double resVal = 0;
        for(int j = 0; j < size; j++) {
            if(row[mtxPtr].col == j && vec.nzValues[vecPtr].col == j) {
                resVal += row[mtxPtr].value * vec.nzValues[vecPtr].value;
                mtxPtr++; vecPtr++;
            }
            else if(row[mtxPtr].col == j)
                mtxPtr++;
            else if(vec.nzValues[vecPtr].col == j)
                vecPtr++;

            //os restantes elementos de um deles sao zero
            if(mtxPtr == row.size() || vecPtr == vec.nzValues.size()) break;
        }
        if(resVal != 0)
            res.insertValue(SparseDouble(i, resVal));
        
    }
    sort(res.nzValues.begin(), res.nzValues.end());
    return res;
}

double dotProduct(vector<double> a, vector<double> b, int begin, int end) {
	double dotProd = 0.0;

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a[i] * b[i]);
	}
    return dotProd;
}

double dotProductSparseVec(vector<SparseDouble> a, vector<SparseDouble> b, int begin, int end) {
    double dotProd = 0;
    int vecPtrA = 0, vecPtrB = 0;

    if(a.size() == 0 || b.size() == 0) 
        return 0;

    for(int i = begin; i < end; i++) {
        if(a[vecPtrA].col == i && b[vecPtrB].col == i) {
            dotProd += a[vecPtrA].value * b[vecPtrB].value;
            vecPtrA++; vecPtrB++;
        }
        else if(a[vecPtrA].col == i)
            vecPtrA++;
        else if(b[vecPtrB].col == i)
            vecPtrB++;
        
        if(vecPtrA == a.size() || vecPtrB == b.size()) break;
    }
    return dotProd;
}