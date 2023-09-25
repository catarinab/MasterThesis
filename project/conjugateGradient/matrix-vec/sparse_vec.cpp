#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;


struct SparseDouble {
    int col;
    int value;
};

bool operator<( const SparseDouble& a, const SparseDouble&b ){
    return a.col < b.col;
}

class Sparse_Vec {

    private:
        int size; //nr colunas
        int nz;
        vector<SparseDouble> nzValues = vector<SparseDouble>();
        omp_lock_t writelock;

    public:

    Sparse_Vec() : size(0) {
        omp_init_lock(&writelock);
    }

    Sparse_Vec(int size) : size(size), nz(0) {
        omp_init_lock(&writelock);
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
            if (value[i] != 0) {
                SparseDouble sd = {col[i], value[i]};
                omp_set_lock(&writelock);
                this->nzValues.push_back(sd);
                this->nz++;
                omp_unset_lock(&writelock);
            }
        }
        sort(this->nzValues.begin(), this->nzValues.end());
    }

    void endRoutine() {
        omp_destroy_lock(&writelock);
    }
};

