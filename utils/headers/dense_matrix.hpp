#ifndef DENSE_MAdoubleRIX_HPP
#define  DENSE_MAdoubleRIX_HPP
#include "dense_vector.hpp"
#include <stdlib.h> //srand
#include <string>
#include <time.h> 

using namespace std;

//A matrix with dense columns (vector of dense vectors)
class dense_matrix {

    private:
        int rows;
        int cols;
        vector<double> values; //for blas

    public:

    dense_matrix(int rows, int cols) : rows(rows), cols(cols) {
        srand((unsigned) time(NULL));
        this->values = vector<double>(rows * cols);
    }

    dense_matrix() : rows(0), cols(0) {
        this->values = vector<double>(rows * cols);
    }

    void setIdentity();
    void setOnesMatrix();
    void setRandomHessenbergMatrix(int minVal, int maxVal);
    void setCol(int col, dense_vector vec);
    void setValue(int row, int col, double val);
    void setValues(vector<double> values);

    double getValue(int row, int col);
    vector<double> getValues();
    dense_vector getCol(int col);
    double* getDataPointer();
    int getRowVal();
    int getColVal();
    double getNorm2();

    dense_matrix mtxAbs();

    void printVals();
    void printMatlab(std::string name);

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- ();
};

#endif