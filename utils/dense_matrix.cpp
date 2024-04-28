#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>

#include "headers/dense_matrix.hpp"
#include "headers/utils.hpp"

string folder("0-1");


using namespace std;

dense_matrix::dense_matrix(int rows, int cols): rows(rows), cols(cols) {
        this->values = vector<double>(rows * cols, 0);
}

int dense_matrix::getRowVal() const {
    return this->rows;
}

int dense_matrix::getColVal() const {
    return this->cols;
}

//get identity matrix
void dense_matrix::setIdentity() {
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++)
        this->setValue(i, i, 1);
}


void dense_matrix::setRandomHessenbergMatrix(double minVal, double maxVal) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    //Create a uniform distribution for random numbers between minVal and maxVal
    uniform_real_distribution<double> distribution(minVal, maxVal);
    for(int row = 0; row < this->rows; row++)
        for(int col = 0; col < this->cols; col++)
            if (row <= col + 1)
                this->setValue(row, col, distribution(generator));
}

void dense_matrix::setRandomUpperTriangularMatrix(double minVal, double maxVal) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    //Create a uniform distribution for random numbers between minVal and maxVal
    uniform_real_distribution<double> distribution(minVal, maxVal);
    for(int row = 0; row < this->rows; row++)
        for(int col = 0; col < this->cols; col++)
            if (row <= col)
                this->setValue(row, col, distribution(generator));

}

void dense_matrix::setRandomMatrix(double minVal, double maxVal) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    //Create a uniform distribution for random numbers between 0 and 1
    std::uniform_real_distribution<double> distribution(minVal, maxVal);
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            this->setValue(i, j, distribution(generator));
        }
    }
}

void dense_matrix::setRandomDiagonalMatrix(double minVal, double maxVal) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // Create a uniform distribution for random numbers between 0 and 1
    std::uniform_real_distribution<double> distribution(minVal, maxVal);
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            if(j == i)
                this->setValue(i, j, distribution(generator));
            else
                this->setValue(i, j, 0);
        }
    }
}


const double* dense_matrix::getDataPointer() const {
        return this->values.data();
}

//insert column in matrix
void dense_matrix::setCol(int col, dense_vector vec){
    for(int row = 0; row < this->rows; row++)
        this->values[row * this->cols + col] = vec.values[row];
}

//insert value in matrix
void dense_matrix::setValue(int row, int col, double val){
    this->values[row * this->cols + col] = val;
}

void dense_matrix::setValues(vector<double> newVals) {
    this->values = std::move(newVals);
}


double dense_matrix::getValue(int row, int col) const{
    return this->values[row * this->cols + col];
}

vector<double> dense_matrix::getValues() {
    return this->values;
}

//get specific column (as a dense_vector)
dense_vector dense_matrix::getCol(int col) {
    dense_vector res(this->rows);

    // Row major order
    for (int row = 0; row < this->rows; row++) {
        res.setValue(row, this->values[row * this->cols + col]);
    }

    return res;
}

//get specific column (in an existing dense_vector)
void dense_matrix::getCol(int col, dense_vector * res) {
    #pragma omp parallel for
    for (int row = 0; row < this->rows; row++) {
        res->setValue(row, this->values[row * this->cols + col]);
    }
}

void dense_matrix::getCol(int col, vector<double> * res){
    #pragma omp for
    for (int row = 0; row < this->rows; row++) {
        res->push_back(this->values[row * this->cols + col]);
    }

}

//get matrix norm2
double dense_matrix::getNorm2() {
    double res = 0;
     for(int i = 0; i < this->rows; i++) {
         for(int j = 0; j < this->cols; j++)
             res += pow(this->values[i * this->cols + j], 2);
    }
    return sqrt(res);
}

void dense_matrix::printVector(const string& filename) {
        ofstream myFile;
        myFile.open(filename);
        if(!myFile.is_open()) {
            cout << "Error opening file" << endl;
            return;
        }
        myFile << this->rows << endl;
        for(double value : this->values) {
            myFile << std::setprecision (16) << value << endl;
        }
        myFile.close();
    }

void dense_matrix::readVector(const string& filename) {
    int count = 0;
    ifstream inputFile(filename);
    if (inputFile) {
        double value;
        inputFile >> value;
        this->rows = (int) value;
        this->cols = (int) value;
        this->values = vector<double>(this->rows * this->cols, 0);
        while (inputFile >> value) {
            this->values[count++] = value;
        }
        inputFile.close();
    }
    else {
        cout << "Error opening file" << endl;
    }

}

void dense_matrix::printMatlabFile(const string& fileName) {
    this->printVector(folder+"vector.txt");
    ofstream myFile;
    myFile.open("/mnt/c/Users/catar/Documents/MATLAB/tese/" + fileName);
    if (myFile) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                myFile << scientific << std::setprecision(16) << this->values[i * this->cols + j] << ",";
            }
            myFile << endl;
        }
    }
    else {
        cout << "Error opening file" << endl;
    }
}

//divide all elements of matrix by x
dense_matrix dense_matrix::operator/ (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->values[i * this->cols + j] / x);

    return res;
}

//multiply all elements of matrix by x
dense_matrix dense_matrix::operator* (double x) {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++) 
            res.setValue(i, j, this->values[i * this->cols + j] * x);

    return res;
}

//subtract all elements of matrix by x
dense_matrix dense_matrix::operator- () const {
    dense_matrix res(this->rows, this->cols);
    #pragma omp parallel for
    for(int i = 0; i < this->rows; i++) 
        for(int j = 0; j < this->cols; j++)
            res.setValue(i, j, -this->values[i * this->cols + j]);
    return res; 
}

void dense_matrix::printMatrix() {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            cout << this->values[i * this->cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

}
