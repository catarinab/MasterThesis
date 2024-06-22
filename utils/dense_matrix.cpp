#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <algorithm>

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

void dense_matrix::setCol(int col, double * ptr, int elements){
    memcpy(this->values + col * this->rows, ptr, elements * sizeof(double));
}

//insert value in matrix
void dense_matrix::setValue(int row, int col, double val){
    this->values[row + col * this->rows] = val;
}


double dense_matrix::getValue(int row, int col) const{
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
#pragma omp parallel for
    for (int row = 0; row < this->rows; row++) {
        res->setValue(row, this->values[row + col * this->rows]);
    }
}

void dense_matrix::getCol(int col, vector<double>& res){
#pragma omp for
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
    double res = 0;
    for(int row = 0; row < this->rows; row++) {
        for(int col = 0; col < this->cols; col++)
            res += pow(this->values[row + col * this->rows], 2);
    }
    return sqrt(res);
}

void dense_matrix::printVector(const string& filename) {
    ofstream myFile;
    myFile.open(filename);
    if(!myFile.is_open()) {
        cout << "Error opening file" << endl;
        return;
    }
    myFile << this->rows << endl;
    for(int i = 0; i < this->rows; i++) {
        double value = this->values[i];
        myFile << std::setprecision (16) << value << endl;
    }
    myFile.close();
}

void dense_matrix::readVector(const string& filename) {
    int count = 0;
    ifstream inputFile(filename);
    if (inputFile) {
        double value;
        inputFile >> value;
        this->rows = (int) value;
        this->cols = (int) value;
        this->values = static_cast<double *>(calloc(this->rows * this->cols, sizeof(double)));
        while (inputFile >> value) {
            this->values[count++] = value;
        }
        inputFile.close();
    }
    else {
        cout << "Error opening file" << endl;
    }

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


void dense_matrix::printMatrix(const string& name) {
    cout <<"Matrix " << name << " : " << this->rows << " x :" << this->cols << endl;
    for(int row = 0; row < this->rows; row++) {
        for(int col = 0; col < this->cols; col++) {
            cout << this->values[row + col * this->rows] << " ";
        }
        cout << endl;
    }
    cout << endl;

}

void dense_matrix::getCol(int col, double *ptr) {
    for(int row = 0; row < this->rows; row++) {
        ptr[row] = this->values[row + col * this->rows];
    }
}

void dense_matrix::setCol(int col, const double *ptr) {
    for(int row = 0; row < this->rows; row++) {
        this->values[row + col * this->rows] = ptr[row];
    }
}

void dense_matrix::setCol(int col, const double *ptr, int changedRows) {
    for(int row = 0; row < changedRows; row++) {
        this->values[row + col * this->rows] = ptr[row];
    }
}

void dense_matrix::checkNanorInf() {
    for(int row = 0; row < this->rows; row++) {
        for(int col = 0; col < this->cols; col++) {
            if(isnan(this->values[row + col * this->rows]) || isinf(this->values[row + col * this->rows])) {
                cout << "Nan or Inf found in matrix" << endl;
                exit(1);
            }
        }
    }
}
