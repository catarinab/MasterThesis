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
        for(int i = 0; i < cols; i++)
            this->columns[i] = Vector(rows);
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

    double getRowVal() {
        return this->rows;
    }

    dense_Matrix getSlice(int finalRow, int finalCol) {
        dense_Matrix res(rows, cols);
        for(int j = 0; j < finalRow; j++) {
            for(int i = 0; i < finalCol; i++) {
                res.setValue(j, i, this->columns[i].values[j]);
            }
        }
        return res;
    }

    void printAttr(string name) {
        cout << "dense matrix: " << name << endl;
        cout << "rows: " << this->rows << endl;
        cout << "cols: " << this->cols << endl;
        cout << "columns: " << endl;
        for(int j = 0; j < this->rows; j++) {
            cout << "row " << j << ": ";
            for(int i = 0; i < this->cols; i++) {
                cout << this->columns[i].values[j] << " ";
            }
            cout << endl;
        }
    }
};