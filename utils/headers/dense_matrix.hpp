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

    dense_matrix(int rows, int cols);

    dense_matrix() : rows(0), cols(0) {
        this->values = vector<double>();
    }

    void setIdentity();
    void setRandomHessenbergMatrix(double minVal, double maxVal);
    void setRandomUpperTriangularMatrix(double minVal, double maxVal);
    void setRandomMatrix(double minVal, double maxVal);
    void setRandomDiagonalMatrix(double minVal, double maxVal);
    void setCol(int col, dense_vector vec);
    void setValue(int row, int col, double val);
    void setValues(vector<double> values);

    [[nodiscard]] double getValue(int row, int col) const;
    vector<double> getValues();
    void getCol(int col, dense_vector * res);
    dense_vector getCol(int col);
    [[nodiscard]] const double* getDataPointer() const;
    [[nodiscard]] int getRowVal() const;
    [[nodiscard]] int getColVal() const;
    double getNorm2();

    void printMatlabFile(const string& fileName);
    void printVector(const string& filename);
    void readVector(const string& currFolder);
    void printMatrix();

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- () const;
};

#endif