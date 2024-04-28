#include <iostream>
#include <utility>
#include <vector>
#include <cmath> //sqrt

#include "headers/dense_vector.hpp"

using namespace std;

int dense_vector::getSize() const {
    return this->size;
}

void dense_vector::resize(int newSize) {
    this->size = newSize;
    this->values.resize(newSize);
}

void dense_vector::getOnesVec() {
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) 
        this->values[i] = 1;
}

void dense_vector::getZeroVec() {
    for(int i = 0; i < this->size; i++)
        this->values[i] = 0;
}

//multiply all vector values by x
dense_vector dense_vector::operator* (double x) {
    dense_vector res(this->size);

    for(int i = 0; i < this->size; i++) {
        double newVal = this->values[i] * x;
        res.insertValue(i, newVal);
    }
    return res;
}

//divide all vector values by x
dense_vector dense_vector::operator/ (double x) {
    dense_vector res(this->size);
    
    for(int i = 0; i < this->size; i++) {
        double newVal = this->values[i] / x;
        res.insertValue(i, newVal);
    }
    return res;
}

void dense_vector::setValues(vector<double> newValues) {
    this->values = std::move(newValues);
}

//insert value in vector
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
        res += abs(this->values[i]) * abs(this->values[i]);
    }
    return (double) sqrt(res);
}

double dense_vector::getValue(int i) {
    return this->values[i];
}

void dense_vector::setValue(int i, double value) {
    this->values[i] = value;
}

std::vector<double> dense_vector::getValues() const {
    return this->values;
}

