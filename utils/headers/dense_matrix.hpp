#ifndef DENSE_MATRIX_HPP
#define  DENSE_MATRIX_HPP
#include "dense_vector.hpp"

using namespace std;

//A matrix with dense columns (array of vectors)
class dense_matrix {

    private:
        int rows;
        int cols;
        dense_vector * columns;

    public:

    dense_matrix(int rows, int cols);
    dense_matrix();

    void setIdentity();
    void setCol(int col, dense_vector vec);
    void setValue(int row, int col, double val);

    double getValue(int row, int col);
    dense_vector getCol(int col);
    dense_vector getRow(int row);
    int getRowVal();
    int getColVal();
    double getNorm2();

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- ();
};

#endif