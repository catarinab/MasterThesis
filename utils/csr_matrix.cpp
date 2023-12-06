#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm> //sort
#include <math.h> //sqrt
#ifndef SPARSE_STRUCT_H
#define SPARSE_STRUCT_H 1
    #include "sparse_structs.h"
#endif


using namespace std;


class CSR_Matrix {

    private:
        int size; //number of rows and columns, only square matrices
        int nz; //number of nonzero entries
        vector<double> nzValues = vector<double>();
        vector<int> colIndex = vector<int>();
        vector<int> rowPtr;

    public:

    CSR_Matrix(int size) : size(size), nz(0) {
        this->rowPtr = vector<int>(size + 1);
    }

    int getNZ() {
        return this->nz;	
    }

    int getSize() {
        return this->size;
    }

    void insertRow(vector<SparseTriplet> row, int rowId) {
        sort(row.begin(), row.end());
        for (int i = 0; i < row.size(); i++) {
            if (row[i].value != 0) {
                this->nzValues.push_back(row[i].value);
                this->colIndex.push_back(row[i].col);
                this->nz++;
            }
        }
        this->rowPtr[rowId + 1] = this->nz;
    }

    vector<SparseTriplet> getRow(int row) {
        vector<SparseTriplet> rowValues = vector<SparseTriplet>();
        int start = this->rowPtr[row];
        int end = this->rowPtr[row + 1];
        for (int i = start; i < end; i++) {
            SparseTriplet triplet(row, this->colIndex[i], this->nzValues[i]);
            rowValues.push_back(triplet);
        }
        return rowValues;
    }

    void getNorm2() {
        double norm = 0;
        for (int i = 0; i < this->nzValues.size(); i++) {
            norm += this->nzValues[i] * this->nzValues[i];
        }
        cout << "norm: " << norm << endl;
    }

    void printAttr() {
        cout << "size: " << this->size << endl;
        cout << "nz: " << this->nz << endl;
        cout << "nzValues: ";
        for (int i = 0; i < this->nzValues.size(); i++) {
            cout << this->nzValues[i] << " ";
        }
        cout << endl;
        cout << "colIndex: ";
        for (int i = 0; i < this->colIndex.size(); i++) {
            cout << this->colIndex[i] << " ";
        }
        cout << endl;
        cout << "rowPtr: ";
        for (int i = 0; i < this->rowPtr.size(); i++) {
            cout << this->rowPtr[i] << " ";
        }
        cout << endl;
    }

};