#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h>

using namespace std;

//A matrix with dense columns (array of vectors)
class dense_Matrix {

    protected:
        int rows;
        int cols;
        DenseVector * columns;

    public:

    dense_Matrix(int rows, int cols) : rows(rows), cols(cols) {
        this->columns = new DenseVector[cols];

        #pragma omp parallel for
        for(int i = 0; i < cols; i++)
            this->columns[i] = DenseVector(rows);
    }

    dense_Matrix() : rows(0), cols(0) {
        this->columns = new DenseVector[0];
    }

    void deleteCols() {
        delete[] columns; // Free the dynamically allocated memory
    }

    void setIdentity() {
        if(this->rows != this->cols) {
            cout << "Error: matrix is not square" << endl;
            return;
        }
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++)
            this->setValue(i, i, 1);
    }

    void setRandomVals(int range) {
        srand(time(0));
        for(int i = 0; i < this->rows; i++)
            for(int j = 0; j < this->cols; j++)
                this->setValue(i, j, (float) rand()/RAND_MAX * range);
    }

    void setCol(int col, DenseVector vec){
        this->columns[col] = vec;
    }

    void setValue(int row, int col, double val){
        this->columns[col].insertValue(row, val);
    }

    double getValue(int row, int col){
        return this->columns[col].values[row];
    }

    double * getValues() {
        double * values = new double[this->rows * this->cols];
        for(int i = 0; i < this->cols; i++)
            #pragma omp parallel for
            for(int j = 0; j < this->rows; j++){
                int index = i * this->rows + j;
                values[index] = this->columns[i].values[j];
            }
        return values;
    }

    DenseVector getCol(int col){
        return this->columns[col];
    }


    DenseVector getRow(int row){
        DenseVector res(this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->cols; i++)
            res.insertValue(i, this->columns[i].values[row]);
        return res;
    }

    int getRowVal() {
        return this->rows;
    }

    int getColVal() {
        return this->cols;
    }

    void setRowVal(int rows) {
        this->rows = rows;
    }

    void setColVal(int cols) {
        this->cols = cols;
    }

    double getNorm1() {
        double res = 0;
        for(int i = 0; i < this->cols; i++) {
            int currSum = 0;
            #pragma omp parallel for reduction(+:currSum)
            for(int j = 0; j < this->rows; j++)
                currSum += abs(this->columns[i].values[j]);
            if(currSum > res) res = currSum;
        }
        return res;
    }

    double getNorm2() {
        double res = 0;
        for(int i = 0; i < this->cols; i++) {
            #pragma omp parallel for reduction(+:res)
            for(int j = 0; j < this->rows; j++)
                res += pow(this->columns[i].values[j], 2);
        }
        return sqrt(res);
    }

    dense_Matrix operator/ (double x) {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] / x);
    
        return res;
    }

    dense_Matrix operator* (double x) {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] * x);
    
        return res;
    }

    // overloaded unary minus (-) operator
    dense_Matrix operator- () {
        dense_Matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, -this->columns[j].values[i]);  
        return res; 
    }

    

    void printAttr(string name) {
        cout << "dense matrix: " << name << endl;
        cout << "rows: " << this->rows << endl;
        cout << "cols: " << this->cols << endl;
        cout << "columns: " << endl;
        for(int j = 0; j < this->rows; j++) {
            cout << "row " << j << ": ";
            for(int i = 0; i < this->cols; i++) {
                cout << this->columns[i].values[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    void printMatlab() {
        cout << "A = [";
        for(int j = 0; j < this->rows; j++) {
            for(int i = 0; i < this->cols; i++) {
                cout << this->columns[i].values[j] << " ";
            }
            if(j != this->rows - 1)
            cout << ";";
        }
        cout << "];\n" << endl;

    }
};