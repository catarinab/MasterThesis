#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>
#include <time.h>
#ifndef SPARSE_STRUCT_H
#define SPARSE_STRUCT_H 1
    #include "sparse_structs.h"
#endif

using namespace std;


class Sparse_Vec {

    private:
        bool random = false;
        bool aux = false;

    public:
        int size; //nr colunas
        int nz;
        vector<SparseDouble> nzValues;
        


    Sparse_Vec() : size(0) , nz(0) {
        this->nzValues = vector<SparseDouble>(0);
    }

    Sparse_Vec(int size) : size(size), nz(0) {
        this->nzValues = vector<SparseDouble>(0);
    }

    void setNzVec(int nz) {
        this->nz = nz;
        this->nzValues.resize(nz);
    }

    void setNZ(int nz) {
        this->nz += nz;
    }

    void addNz(int nz) {
        this->nz += nz;
        this->nzValues.resize(this->nz);
    }

    Sparse_Vec(int size, bool random) : size(size), nz(0), random(random) {
        //srand(time(0));
        if(random)
            getRandomVec(size);
        else
            this->nzValues = vector<SparseDouble>();
    }

    Sparse_Vec operator-()  {
        Sparse_Vec res(this->size);
        for(int i = 0; i < this->nzValues.size(); i++) {
            res.nzValues.push_back(SparseDouble(this->nzValues[i].col, -this->nzValues[i].value));
        }
        return res;
    }

    Sparse_Vec operator* (double x) {
        Sparse_Vec res(this->size);
        for(int i = 0; i < this->nzValues.size(); i++) {
            double newVal = this->nzValues[i].value * x;
            res.nzValues.push_back(SparseDouble(this->nzValues[i].col, newVal));
        }
        return res;
    }

    void insertValue(SparseDouble sd) {
        #pragma omp critical
        {
            this->nzValues.push_back(sd);
            this->nz++;
        }
    }

    void pushVal(SparseDouble sd) {
        #pragma omp critical
        this->nzValues.push_back(sd);
    }

    //so para testes
    void getRandomVec(int size) {
        for(int i = 0; i < size; i++) {
            bool zeroVal = (rand() % 100) < 90;
            if(!zeroVal) {
                double val = (rand() % 100);
                insertValue(SparseDouble(i, val));
            }
        }
        sort(this->nzValues.begin(), this->nzValues.end());
    }

    void printAttr() {
        cout << "size: " << this->size << endl;
        cout << "nz: " << this->nz << endl;
        cout << "nzValues: ";
        for (int i = 0; i < this->nzValues.size(); i++) {
            cout << this->nzValues[i].value << " ";
        }
        cout << endl;
        cout << "colIndex: ";
        for (int i = 0; i < this->nzValues.size(); i++) {
            cout << this->nzValues[i].col << " ";
        }
        cout << endl;
    }

    void insertValues(int col[], int value[]) {
        //acho que nao vale a pena paralelizar (?)
        #pragma omp parallel for
        for (int i = 0; i < this->size; i++) {
            if (value[i] != 0) 
                insertValue(SparseDouble(col[i], value[i]));
        }
        sort(this->nzValues.begin(), this->nzValues.end());
    }
};

