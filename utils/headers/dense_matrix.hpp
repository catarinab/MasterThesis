#ifndef DENSE_MATRIX_HPP
#define  DENSE_MATRIX_HPP
#include "dense_vector.hpp"
#include <cstdlib> //srand
#include <string>
#include <ctime>

using namespace std;

//A matrix with dense columns (vector of dense vectors)
class dense_matrix {

    private:
        int rows;
        int cols;
        vector<double> values; //for blas

    public:

    dense_matrix(int rows, int cols) : rows(rows), cols(cols) {
        srand((unsigned) time(nullptr));
        this->values = vector<double>(rows * cols);
    }

    dense_matrix() : rows(0), cols(0) {
        this->values = vector<double>(rows * cols);
    }

    void setIdentity();
    void setOnesMatrix();
    void setRandomHessenbergMatrix(int minVal, int maxVal);
    void setRandomUpperTriangularMatrix(int minVal, int maxVal);
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
    void printMatlab(const std::string& name);

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- ();
};

#endif