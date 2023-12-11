#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <iostream>
#include <vector>
#include <algorithm> //sort
#include <math.h>    //sqrt
#include "sparse_structs.h"

using namespace std;

class csr_matrix {

private:
    int size; //number of rows and columns, only square matrices
    int nz;  //number of nonzero entries
    vector<double> nzValues;
    vector<int> colIndex;
    vector<int> rowPtr;

public:
    inline csr_matrix(int size) : size(size), nz(0), rowPtr(vector<int>(size + 1)) {}
    int getNZ();
    int getSize();
    void insertRow(vector<SparseTriplet> row, int rowId);
    vector<SparseTriplet> getRow(int row);
    void getNorm2();
    void printAttr();
};

#endif // CSR_MATRIX_HPP
