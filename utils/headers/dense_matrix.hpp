#ifndef DENSE_MATRIX_HPP
#define  DENSE_MATRIX_HPP
#include "dense_vector.hpp"

using namespace std;

//A matrix with dense columns (vector of dense vectors)
class dense_matrix {

    private:
        int rows;
        int cols;
        vector<double> values; //for blas

    public:

    dense_matrix(int rows, int cols);
    dense_matrix();

    void setIdentity();
    void setRandomMatrix();
    void setCol(int col, dense_vector vec);
    void setValue(int row, int col, double val);

    double getValue(int row, int col);
    dense_vector getCol(int col);
    double* getDataPointer();
    int getRowVal();
    int getColVal();
    double getNorm2();

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- ();
};

#endif