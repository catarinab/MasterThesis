#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>

#include "headers/dense_matrix.hpp"

using namespace std;

//A matrix with dense columns (vector of dense vectors)
dense_matrix::dense_matrix(int rows, int cols) : rows(rows), cols(cols) {
    this->columns = vector<dense_vector>(cols);
    #pragma omp parallel for
    for(int i = 0; i < cols; i++)
        this->columns[i] = dense_vector(rows);
}

dense_matrix::dense_matrix() : rows(0), cols(0) {
    this->columns = vector<dense_vector>(cols);
}


int dense_matrix::getRowVal() {
    return this->rows;
}

int dense_matrix::getColVal() {
    return this->cols;
}

//get identity matrix
void dense_matrix::setIdentity() {
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++)
        this->setValue(i, i, 1);
}

//insert column in matrix
void dense_matrix::setCol(int col, dense_vector vec){
    this->columns[col] = vec;
}

//insert value in matrix
void dense_matrix::setValue(int row, int col, double val){
    this->columns[col].insertValue(row, val);
}

double dense_matrix::getValue(int row, int col){
    return this->columns[col].values[row];
}

//get spefic column (as a dense_vector)
dense_vector dense_matrix::getCol(int col){
    return this->columns[col];
}

//get specific row (as a dense_vector)
dense_vector dense_matrix::getRow(int row){
    dense_vector res(this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->cols; i++)
        res.insertValue(i, this->columns[i].values[row]);
    return res;
}

//get matrix norm2
double dense_matrix::getNorm2() {
    double res = 0;
    for(int i = 0; i < this->cols; i++) {
        #pragma omp parallel for reduction(+:res)
        for(int j = 0; j < this->rows; j++)
            res += pow(this->columns[i].values[j], 2);
    }
    return sqrt(res);
}

//divide all elements of matrix by x
dense_matrix dense_matrix::operator/ (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->columns[j].values[i] / x);

    return res;
}

//multiply all elements of matrix by x
dense_matrix dense_matrix::operator* (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->columns[j].values[i] * x);

    return res;
}

//subtract all elements of matrix by x
dense_matrix dense_matrix::operator- () {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, -this->columns[j].values[i]);  
    return res; 
}
