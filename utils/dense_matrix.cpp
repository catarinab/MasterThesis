#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>

#include "headers/dense_matrix.hpp"

using namespace std;

//A matrix with dense columns (vector of dense vectors)
dense_matrix::dense_matrix(int rows, int cols) : rows(rows), cols(cols) {
    this->values = vector<double>(rows * cols);
}

dense_matrix::dense_matrix() : rows(0), cols(0) {
    this->values = vector<double>(rows * cols);
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

void dense_matrix::setRandomMatrix() {
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++)
        for(int j = 0; j < this->cols; j++)
            this->setValue(i, j, rand() % 10);
}

double* dense_matrix::getDataPointer() {
        return this->values.data();
}

//insert column in matrix
void dense_matrix::setCol(int col, dense_vector vec){
    for(int i = 0; i < this->rows; i++)
        this->values[col * this->rows + i] = vec.values[i];
}

//insert value in matrix
void dense_matrix::setValue(int row, int col, double val){
    this->values[col * this->rows + row] = val;
}

void dense_matrix::setValues(vector<double> values) {
    this->values = values;
}


double dense_matrix::getValue(int row, int col){
    return this->values[col * this->rows + row];
}

//get spefic column (as a dense_vector)
dense_vector dense_matrix::getCol(int col){
    dense_vector res(this->rows);
    
    res.setValues(vector<double>(this->values.begin() + col * this->rows, 
                                this->values.begin() + (col + 1) * this->rows));
    return res;
}

//get matrix norm2
double dense_matrix::getNorm2() {
    double res = 0;
    for(int i = 0; i < this->cols; i++) {
        #pragma omp parallel for reduction(+:res)
        for(int j = 0; j < this->rows; j++)
            res += pow(this->values[i * this->rows + j], 2);
    }
    return sqrt(res);
}

//divide all elements of matrix by x
dense_matrix dense_matrix::operator/ (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->values[j * this->rows + i] / x);

    return res;
}

//multiply all elements of matrix by x
dense_matrix dense_matrix::operator* (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->values[j * this->rows + i] * x);

    return res;
}

//subtract all elements of matrix by x
dense_matrix dense_matrix::operator- () {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, -this->values[j * this->rows + i]);  
    return res; 
}
