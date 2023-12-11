#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h> //sqrt

#include "headers/dense_vector.hpp"

using namespace std;

int dense_vector::getSize() {
    return this->size;
}

void dense_vector::setValues(vector<double> values) {
    this->size = size;
    this->values = values;
}


void dense_vector::resize(int size) {
    this->size = size;
    this->values.resize(size);
}

//so para testes
void dense_vector::getRandomVec() {
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) {
        double val = (rand() % 100);
        this->values[i] = val;
    }
}

void dense_vector::getOnesVec() {
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) 
        this->values[i] = 1;
}

dense_vector dense_vector::operator* (double x) {
    dense_vector res(this->size);
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) {
        double newVal = this->values[i] * x;
        res.insertValue(i, newVal);
    }
    return res;
}

dense_vector dense_vector::operator/ (double x) {
    dense_vector res(this->size);
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) {
        double newVal = this->values[i] / x;
        res.insertValue(i, newVal);
    }
    return res;
}

void dense_vector::insertValue(int col, double value) {
    if(col >= this->size) {
        cout << "col: " << col << endl;
        cout << "size: " << this->size << endl;
        cout << "error: col out of bounds" << endl;
        exit(1);
    }
    this->values[col] = value;
}

double dense_vector::getNorm2() {
    double res = 0;

    #pragma omp parallel for reduction(+:res)
    for(int i = 0; i < this->size; i++) {
        res += this->values[i] * this->values[i];
    }
    return (double) sqrt(res);
}

void dense_vector::printAttr(string name) {
    cout << name  << ":" << endl;
    cout << "size: " << this->size << endl;
    cout << "values: " << endl;
    for (int i = 0; i < size; i++) {
        cout << this->values[i] << ", ";
    }
    cout << "\n" << endl;
}

