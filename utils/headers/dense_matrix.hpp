#ifndef DENSE_MATRIX_HPP
#define  DENSE_MATRIX_HPP
#include "dense_vector.hpp"
#include <cstdlib> //srand
#include <string>
#include <ctime>
#include <cstring>

using namespace std;

//A Column Major Dense Matrix
class dense_matrix {

    private:
        int rows;
        int cols;
        double * values;

    public:

    dense_matrix(int rows, int cols): rows(rows), cols(cols) {
        this->values = static_cast<double *>(calloc(rows * cols, sizeof(double)));
    }

    dense_matrix() : rows(0), cols(0) {
        this->values = static_cast<double *>(calloc(1, sizeof(double)));
    }

    dense_matrix(const dense_matrix& other) : rows(other.rows), cols(other.cols) {
        this->values = this->values = static_cast<double *>(calloc(rows * cols, sizeof(double)));
        std::copy(other.values, other.values + rows * cols, this->values);
    }

    dense_matrix& operator=(const dense_matrix& other) {
        if (this != &other) {
            delete[] values;
            rows = other.rows;
            cols = other.cols;
            values = new double[rows * cols];
            std::copy(other.values, other.values + rows * cols, values);
        }
        return *this;
    }

    ~dense_matrix(){
        free(this->values);
    }

    void setIdentity();
    void setCol(int col, dense_vector& vec);
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

    bool hasNanorInf();

    double *getValues();

    void getCol(int col, double **ptr, int startingRow);
};

#endif