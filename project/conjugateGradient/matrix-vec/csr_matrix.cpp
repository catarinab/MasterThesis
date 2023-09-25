#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
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
        int size; //nr rows e de colunas
        int nz;
        vector<double> nzValues = vector<double>();
        vector<int> colIndex = vector<int>();
        vector<int> rowPtr;

    public:

    CSR_Matrix() : size(0) {
        this->rowPtr = vector<int>(0);
    }

    CSR_Matrix(int size) : size(size), nz(0) {
        this->rowPtr = vector<int>(size + 1);
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

    void insertRow(SparseTriplet row[]) {
        //implementar parallel sort ?
        //sort(row, row + this->size);
        //como paralelizar ?????
        int currRow = row[0].row;
        for (int i = 0; i < this->size; i++) {
            if (row[i].value != 0) {
                this->nzValues.push_back(row[i].value);
                this->colIndex.push_back(row[i].col);
                this->nz++;
            }
        }
        this->rowPtr[currRow + 1] = this->nz;
    }

    vector<SparseTriplet> getRow(int row) {
        vector<SparseTriplet> rowValues = vector<SparseTriplet>();
        int start = this->rowPtr[row];
        int end = this->rowPtr[row + 1];
        for (int i = start; i < end; i++) {
            SparseTriplet triplet;
            triplet.row = row;
            triplet.col = this->colIndex[i];
            triplet.value = this->nzValues[i];
            rowValues.push_back(triplet);
        }
        return rowValues;
    }
};