#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h> //sqrt

#include "headers/dense_vector.hpp"

using namespace std;

int dense_vector::getSize() {
    return this->size;
}

void dense_vector::resize(int size) {
    this->size = size;
    this->values.resize(size);
}

void dense_vector::getOnesVec() {
    #pragma omp parallel for
    for(int i = 0; i < this->size; i++) 
        this->values[i] = 1;
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
        res += this->values[i] * this->values[i];
    }
    return (double) sqrt(res);
}

