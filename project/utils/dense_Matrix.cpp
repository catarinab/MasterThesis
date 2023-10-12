#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>


using namespace std;

//A matrix with dense columns (array of vectors)
class dense_Matrix {

    private:
        int rows;
        int cols;
        Vector * columns;

    public:

    dense_Matrix(int rows, int cols) : rows(rows), cols(cols) {
        this->columns = new Vector[cols];
        for(int i = 0; i < rows; i++)
            this->columns[i].resize(rows);
    }

    void setCol(int col, Vector vec){
        this->columns[col] = vec;
    }

    void setValue(int row, int col, double val){
        this->columns[col].insertValue(row, val);
    }

    double getValue(int row, int col){
        return this->columns[col].values[row];
    }

    Vector getCol(int col){
        return this->columns[col];
    }

    void printAttr() {
        cout << "rows: " << this->rows << endl;
        cout << "cols: " << this->cols << endl;
        cout << "columns: " << endl;
        for (int i = 0; i < this->cols; i++) {
            cout << "col " << i << ": ";
            for (int j = 0; j < this->rows; j++) {
                cout << this->columns[i].values[j] << " ";
            }
            cout << endl;
        }
    }
};