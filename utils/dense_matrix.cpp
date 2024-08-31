#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <mkl.h>

#include "headers/dense_matrix.hpp"

using namespace std;

//A Column Major Dense Matrix
int dense_matrix::getRowVal() const {
    return this->rows;
}

int dense_matrix::getColVal() const {
    return this->cols;
}

//get identity matrix
void dense_matrix::setIdentity() {
    for(int i = 0; i < this->rows; i++) {
        this->values[i + i * this->rows] = 1;
    }
}

void dense_matrix::resize(int size) {
    this->rows = size;
    this->cols = size;
}

void dense_matrix::resizeCols(int newCols) {
    this->cols = newCols;
}

const double* dense_matrix::getDataPointer() const {
    return this->values;
}

//insert column in matrix
void dense_matrix::setCol(int col, dense_vector& vec){
    memcpy(this->values + col * this->rows, vec.values.data(), this->rows * sizeof(double));
}

void dense_matrix::setCol(int col, dense_vector& vec, int start, int count){
    memcpy(this->values + col * this->rows, vec.values.data() + start, count * sizeof(double));
}

//insert value in matrix
void dense_matrix::setValue(int row, int col, double val){
    if(row >= this->rows || col >= this->cols) {
        return;
    }
    this->values[row + col * this->rows] = val;
}


double dense_matrix::getValue(int row, int col) const{
    if(row >= this->rows || col >= this->cols || row < 0 || col < 0) {
        //cerr << "getVal, Error: Index out of bounds" << endl;
        return 0;
    }
    return this->values[row + col * this->rows];
}

double * dense_matrix::getValues() {
    return this->values;
}

//get specific column (as a dense_vector)
dense_vector dense_matrix::getCol(int col) {
    dense_vector res(this->rows);
    // Row major order
    for (int row = 0; row < this->rows; row++) {
        res.setValue(row, this->values[row + col * this->rows]);
    }

    return res;
}

//get specific column (in an existing dense_vector)
void dense_matrix::getCol(int col, dense_vector * res) {
    for (int row = 0; row < this->rows; row++) {
        res->setValue(row, this->values[row + col * this->rows]);
    }
}

void dense_matrix::getCol(int col, vector<double>& res){
    for (int row = 0; row < this->rows; row++) {
        res[row] = this->values[row + col * this->rows];
    }
}

void dense_matrix::getCol(int col, double **ptr) {
    *ptr = this->values + col * this->rows;
}

void dense_matrix::getCol(int col, double ** ptr, int startingRow) {
    *ptr = this->values + startingRow + col * this->rows;
}

//get matrix norm2
double dense_matrix::getNorm2() {
    return cblas_dnrm2(this->rows * this->cols, this->values, 1);
}

//divide all elements of matrix by x
dense_matrix dense_matrix::operator/ (double x) {
    dense_matrix res(this->rows, this->cols);

    for(int row = 0; row < this->rows; row++)
        for(int col = 0; col < this->cols; col++)
            res.setValue(row, col, this->values[row + col * this->rows] / x);

    return res;
}

//multiply all elements of matrix by x
dense_matrix dense_matrix::operator* (double x) {
    dense_matrix res(this->rows, this->cols);

    for(int row = 0; row < this->rows; row++)
        for(int col = 0; col < this->cols; col++)
            res.setValue(row, col, this->values[row + col * this->rows] * x);

    return res;
}

dense_matrix dense_matrix::operator- () const {
    dense_matrix res(this->rows, this->cols);
    for(int row = 0; row < this->rows; row++)
        for(int col = 0; col < this->cols; col++)
            res.setValue(row, col, -this->values[row + col * this->rows]);
    return res;
}

void dense_matrix::getCol(int col, double *ptr) {
    for(int row = 0; row < this->rows; row++) {
        ptr[row] = this->values[row + col * this->rows];
    }
}

void dense_matrix::getLastCol(dense_vector &b) {
    for(int row = 0; row < this->rows; row++) {
        b.setValue(row, this->values[row + (this->cols - 1) * this->rows]);
    }

}
