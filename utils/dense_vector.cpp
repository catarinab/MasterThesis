#include <iostream>
#include <utility>
#include <vector>
#include <cmath> //sqrt
#include <mkl.h>

#include "headers/dense_vector.hpp"

using namespace std;

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

//divide all vector values by x
dense_vector dense_vector::operator/= (double x) {
    for(int i = 0; i < this->size; i++) {
        this->values[i] /= x;
    }
    return *this;
}

dense_vector dense_vector::operator-(const dense_vector& other) const {
    dense_vector result(this->size);
    for (int i = 0; i < this->size; i++) {
        result.insertValue(i, this->values[i] - other.values[i]);
    }
    return result;
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
    return cblas_dnrm2(this->size, this->values.data(), 1);
}

void dense_vector::setValue(int i, double value) {
    this->values[i] = value;
}

std::ostream& operator<<(std::ostream& os, const dense_vector& dv) {
    for (double value : dv.values) {
        os << value << endl;
    }
    return os;
}

