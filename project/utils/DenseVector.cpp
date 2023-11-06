#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h>

using namespace std;


class DenseVector {

    public:
        int size; //nr colunas
        vector<double> values;
        


    DenseVector() : size(0){
        this->values = vector<double>(0);
    }    

    DenseVector(int size) : size(size) {
        this->values = vector<double>(size);
    }

    int getSize() {
        return this->size;
    }

    void setValues(vector<double> values) {
        this->size = size;
        this->values = values;
    }


    void resize(int size) {
        this->size = size;
        this->values.resize(size);
    }

    //so para testes
    void getRandomVec() {
        #pragma omp parallel for
        for(int i = 0; i < this->size; i++) {
            double val = (rand() % 100);
            this->values[i] = val;
        }
    }

    void getOnesVec() {
        #pragma omp parallel for
        for(int i = 0; i < this->size; i++) 
            this->values[i] = 1;
    }

    DenseVector operator* (double x) {
        DenseVector res(this->size);
        #pragma omp parallel for
        for(int i = 0; i < this->size; i++) {
            double newVal = this->values[i] * x;
            res.insertValue(i, newVal);
        }
        return res;
    }

    DenseVector operator/ (double x) {
        DenseVector res(this->size);
        #pragma omp parallel for
        for(int i = 0; i < this->size; i++) {
            double newVal = this->values[i] / x;
            res.insertValue(i, newVal);
        }
        return res;
    }

    void insertValue(int col, double value) {
        if(col >= this->size) {
            cout << "col: " << col << endl;
            cout << "size: " << this->size << endl;
            cout << "error: col out of bounds" << endl;
            exit(1);
        }
        this->values[col] = value;
    }

    double getNorm2() {
        double res = 0;

        #pragma omp parallel for reduction(+:res)
        for(int i = 0; i < this->size; i++) {
            res += this->values[i] * this->values[i];
        }
        return (double) sqrt(res);
    }

    void printAttr(string name) {
        cout << name  << ":" << endl;
        cout << "size: " << this->size << endl;
        cout << "values: " << endl;
        for (int i = 0; i < size; i++) {
            cout << this->values[i] << ", ";
        }
        cout << "\n" << endl;
    }
};

