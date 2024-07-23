#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <iostream>
#include <vector>
#include <algorithm> //sort
#include <cmath>    //sqrt
#include <mkl.h>
#include "utils.hpp"


using namespace std;

class csr_matrix {

private:
    long long int size; //number of rows and columns, only square matrices
    long long int nz;  //number of nonzero entries
    vector<double> nzValues;
    vector<long long int> colIndex;
    vector<long long int> rowPtr;
    vector<long long int> pointerB;
    vector<long long int> pointerE;
    sparse_matrix_t mklSparseMatrix;
    matrix_descr mklDescription = {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

public:
    inline explicit csr_matrix(long long int size) : size(size), nz(0), rowPtr(vector<long long int>(size + 1)) {}
    inline explicit csr_matrix() : size(0), nz(0) {}
    [[nodiscard]] long long int getNZ() const;
    [[nodiscard]] long long int getSize() const;
    void insertRow(vector<SparseTriplet> row, int rowId);
    vector<SparseTriplet> getRow(int row);
    void printAttr() const;
    [[nodiscard]] double getValue(int row, int col) const;
    int * getRowPtr() const;

    void defineMKLSparseMatrix();
    [[nodiscard]] sparse_matrix_t getMKLSparseMatrix() const;
    [[nodiscard]] sparse_matrix_t * getMKLSparseMatrixPointer();
    [[nodiscard]] matrix_descr getMKLDescription() const;

    void saveMatrixMarketFile(string & filename);
};

#endif // CSR_MATRIX_HPP