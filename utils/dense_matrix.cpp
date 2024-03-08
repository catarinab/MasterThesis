#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

#include "headers/dense_matrix.hpp"

using namespace std;

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

void dense_matrix::setOnesMatrix() {
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++)
        for(int j = 0; j < this->cols; j++)
            this->setValue(i, j, 1);
}

void dense_matrix::setRandomHessenbergMatrix(int minVal, int maxVal) {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            if (j + 1 >= i) {
                this->setValue(i, j, minVal + rand() % (maxVal - minVal + 1));
            } else {
                this->setValue(i, j, 0);
            }
        }
    }
 }

 //debugging purposes
void dense_matrix::setRandomUpperTriangularMatrix(int minVal, int maxVal) {
     int counter = 0;
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            if(j == i)
                this->setValue(i, j, counter++);
            else if (j >= i) {
                this->setValue(i, j, minVal + rand() % (maxVal - minVal + 1));
            } else {
                this->setValue(i, j, 0);
            }
        }
    }
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

void dense_matrix::setValues(vector<double> newVals) {
    this->values = newVals;
}


double dense_matrix::getValue(int row, int col){
    return this->values[col * this->rows + row];
}

vector<double> dense_matrix::getValues() {
    return this->values;
}

//get specific column (as a dense_vector)
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
        for(int j = 0; j < this->rows; j++)
            res += pow(this->values[i * this->rows + j], 2);
    }
    return sqrt(res);
}

void dense_matrix::printVals() {
    for(int row = 0; row < this->rows; row++) {
        cout << "Row " << row << ": ";
        for(int col = 0; col < this->cols; col++) {
            cout << this->values[col * this->rows + row] << " ";
        }
        cout << endl;
    }
}

void dense_matrix::printMatlab(const string& name) {
    cout << name << " = [";
    for(int row = 0; row < this->rows; row++) {
        for(int col = 0; col < this->cols; col++) {
            if(col != 0)
                cout << ", ";
            std::cout << std::setprecision (15) << this->values[col * this->rows + row];
        }
        if(row != this->rows - 1)
            cout << "; ";
    }
    cout << "];" << endl;
}

dense_matrix dense_matrix::mtxAbs(){
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
        for(int i = 0; i < this->rows; i++)
            for(int j = 0; j < this->cols; j++)
                res.setValue(i, j, abs(this->values[j * this->rows + i]));

        return res;
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
