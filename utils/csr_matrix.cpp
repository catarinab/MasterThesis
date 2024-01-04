#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm> //sort
#include <math.h> //sqrt


#include "headers/sparse_structs.h"
#include "headers/csr_matrix.hpp"

using namespace std;

int csr_matrix::getNZ() {
    return this->nz;	
}

int csr_matrix::getSize() {
    return this->size;
}

//insert row in the csr matrix, used when initializing the matrix
void csr_matrix::insertRow(vector<SparseTriplet> row, int rowId) {
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

//get values from a single row
vector<SparseTriplet> csr_matrix::getRow(int row) {
    vector<SparseTriplet> rowValues = vector<SparseTriplet>();
    int start = this->rowPtr[row];
    int end = this->rowPtr[row + 1];
    for (int i = start; i < end; i++) {
        SparseTriplet triplet(row, this->colIndex[i], this->nzValues[i]);
        rowValues.push_back(triplet);
    }
    return rowValues;
}