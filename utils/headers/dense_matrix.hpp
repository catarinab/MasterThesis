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
        this->values = nullptr;
    }

    dense_matrix(const dense_matrix& other) : rows(other.rows), cols(other.cols) {
        this->values = static_cast<double *>(malloc(rows * cols * sizeof(double)));
        memcpy(this->values, other.values, rows * cols * sizeof(double));
    }

    dense_matrix& operator=(const dense_matrix& other) {
        if (this != &other) {
            free(values);
            this->rows = other.rows;
            this->cols = other.cols;
            this->values = static_cast<double *>(malloc(rows * cols * sizeof(double)));
            memcpy(this->values, other.values, rows * cols * sizeof(double));
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

    dense_matrix operator/ (double x);
    dense_matrix operator* (double x);
    dense_matrix operator- () const;
    void getCol(int col, double ** ptr);

    double *getValues();

    void getCol(int col, double **ptr, int startingRow);

    void setCol(int col, dense_vector &vec, int start, int count);

    void getLastCol(dense_vector & b);

    void resize(int size);

    void resizeCols(int cols);
};

#endif