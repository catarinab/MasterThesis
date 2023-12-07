#include <iostream>
#include <omp.h>
#include <math.h>

using namespace std;

//A matrix with dense columns (array of vectors)
class dense_matrix {

    private:
        int rows;
        int cols;
        dense_vector * columns;

    public:

    dense_matrix(int rows, int cols) : rows(rows), cols(cols) {
        this->columns = new dense_vector[cols];

        #pragma omp parallel for
        for(int i = 0; i < cols; i++)
            this->columns[i] = dense_vector(rows);
    }

    dense_matrix() : rows(0), cols(0) {
        this->columns = new dense_vector[0];
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

    void setCol(int col, dense_vector vec){
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

    dense_vector getCol(int col){
        return this->columns[col];
    }


    dense_vector getRow(int row){
        dense_vector res(this->cols);
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

    dense_matrix operator/ (double x) {
        dense_matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] / x);
    
        return res;
    }

    dense_matrix operator* (double x) {
        dense_matrix res(this->rows, this->cols);
        #pragma omp parallel for
        for(int i = 0; i < this->rows; i++) 
            for(int j = 0; j < this->cols; j++) 
                res.setValue(i, j, this->columns[j].values[i] * x);
    
        return res;
    }

    // overloaded unary minus (-) operator
    dense_matrix operator- () {
        dense_matrix res(this->rows, this->cols);
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