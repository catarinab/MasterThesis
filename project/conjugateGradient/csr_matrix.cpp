#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;


struct SparseTriplet {
    int col;
    int row;
    int value;
};

bool operator<( const SparseTriplet& a, const SparseTriplet&b ){
    return a.col < b.col;
}

class CSR_Matrix {

    private:
        int size;
        int nz;
        vector<double> nzValues;
        vector<int> colIndex;
        vector<int> rowPtr;

    public:

    CSR_Matrix(int size) {
        this->size = size;
        this->nz = 0;
        this->nzValues = vector<double>();
        this->colIndex = vector<int>();
        this->rowPtr = vector<int>(size + 1);
    }

    void insertRow(SparseTriplet row[]) {
        //implementar parallel sort ?
        sort(row, row + this->size);
        //como paralelizar ?????
        for (int i = 0; i < this->size; i++) {
            if (row[i].value != 0) {
                this->nzValues.push_back(row[i].value);
                this->colIndex.push_back(row[i].col);
                this->nz++;
            }
        this->rowPtr[row[i].row + 1] = this->nz;
        }
    }

    
    void insertVal(int row, int col, double value) {
        
    }
};