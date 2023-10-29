#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>


using namespace std;

//A matrix with dense columns (array of vectors)
class dense_Matrix {

    protected:
        int rows;
        int cols;
        Vector * columns;

    public:

    dense_Matrix(int rows, int cols) : rows(rows), cols(cols) {
        this->columns = new Vector[cols];

        #pragma omp parallel for
        for(int i = 0; i < cols; i++)
            this->columns[i] = Vector(rows);
    }

    dense_Matrix() : rows(0), cols(0) {
        this->columns = new Vector[0];
    }

    void setIdentity() {
        if(this->rows != this->cols) {
            cout << "Error: matrix is not square" << endl;
            return;
        }
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++)
            this->setValue(i, i, 1);
    }

    void setRandomSmall() {
        for(int i = 0; i < this->rows; i++)
            for(int j = 0; j < this->cols; j++)
                this->setValue(i, j, (float) rand()/RAND_MAX);
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


    Vector getRow(int row){
        Vector res(this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->cols; i++)
            res.insertValue(i, this->columns[i].values[row]);
        return res;
    }

    int getRowVal() {
        return this->rows;
    }

    int getColVal() {
        return this->cols;
    }

    void setRowVal(int rows) {
        this->rows = rows;
    }

    void setColVal(int cols) {
        this->cols = cols;
    }

    double getNorm1() {
        double res = 0;
        for(int i = 0; i < this->cols; i++) {
            int currSum = 0;
            #pragma omp parallel for reduction(+:currSum)
            for(int j = 0; j < this->rows; j++)
                currSum += abs(this->columns[i].values[j]);
            if(currSum > res) res = currSum;
        }
        return res;
    }

    double getNorm2() {
        double res = 0;
        for(int i = 0; i < this->cols; i++) {
            #pragma omp parallel for reduction(+:res)
            for(int j = 0; j < this->rows; j++)
                res += pow(this->columns[i].values[j], 2);
        }
        return sqrt(res);
    }

    dense_Matrix operator/ (double x) {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] / x);
    
        return res;
    }

    dense_Matrix operator* (double x) {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] * x);
    
        return res;
    }

    // overloaded unary minus (-) operator
    dense_Matrix operator- () {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, -this->columns[j].values[i]);  
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