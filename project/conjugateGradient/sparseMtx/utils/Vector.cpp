#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>
#include <time.h>

using namespace std;


class Vector {

    public:
        int size; //nr colunas
        vector<double> values;
        bool init = false;
        


    Vector() : size(0){
        this->values = vector<double>(0);
        this->init = false;
    }

    Vector(int size, bool random) : size(size) {
        this->values = vector<double>(size);
        this->init = false;
        if(random)
            getRandomVec(size);
    }

    Vector(int size) : size(size) {
        this->values = vector<double>(size);
    }

    void resize(int size) {
        this->size = size;
        this->values.resize(size);
        this->init = false;
    }

    //so para testes
    void getRandomVec(int size) {
        for(int i = 0; i < size; i++) {
            double val = (rand() % 100);
            this->values[i] = val;
        }
        this->init = true;
    }

    Vector operator* (double x) {
        Vector res(this->size);
        for(int i = 0; i < this->size; i++) {
            double newVal = this->values[i] * x;
            res.insertValue(i, newVal);
        }
        return res;
    }

    void insertValue(int col, double value) {
        this->values[col] = value;
    }

    vector<double> getSlice(int begin, int end) {
        vector<double> res(end - begin);
        for(int i = begin; i < end; i++) {
            res[i - begin] = this->values[i];
        }
        return res;
    }

    void printAttr(string name) {
        cout << name  << ":" << endl;
        cout << "size: " << this->size << endl;
        cout << "values: " << endl;
        for (int i = 0; i < size; i++) {
            cout << i << ": " << this->values[i] << endl;
        }
    }
};

