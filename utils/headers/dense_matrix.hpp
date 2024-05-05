#ifndef DENSE_MATRIX_HPP
#define  DENSE_MATRIX_HPP
#include "dense_vector.hpp"
#include <cstdlib> //srand
#include <string>
#include <ctime>

using namespace std;

//A Column Major Dense Matrix
class dense_matrix {

    private:
        int rows;
        int cols;
        vector<double> values;

    public:

    dense_matrix(int rows, int cols);

    dense_matrix() : rows(0), cols(0) {
        this->values = vector<double>();
    }

    void setIdentity();
    void setCol(int col, dense_vector vec);
    void setValue(int row, int col, double val);

    [[nodiscard]] double getValue(int row, int col) const;

    void getCol(int col, double * ptr);
    void getCol(int col, dense_vector * res);
    void getCol(int col, vector<double>& vect);
    dense_vector getCol(int col);

    [[nodiscard]] const double* getDataPointer() const;
    [[nodiscard]] int getRowVal() const;
    [[nodiscard]] int getColVal() const;
    double getNorm2();

    void printVector(const string& filename);
    void readVector(const string& currFolder);
    void printMatrix();

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- () const;
    void getCol(int col, double ** ptr);
};

#endif