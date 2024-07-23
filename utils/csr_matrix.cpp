#include "headers/csr_matrix.hpp"
#include <fstream>

using namespace std;


long long int csr_matrix::getSize() const {
    return this->size;
}

long long int csr_matrix::getNZ() const {
    return this->nz;
}

void csr_matrix::defineMKLSparseMatrix() {
    this->pointerB = vector<long long int>(this->rowPtr.begin(), this->rowPtr.end() - 1);
    this->pointerE = vector<long long int>(this->rowPtr.begin() + 1, this->rowPtr.end());

    mkl_sparse_d_create_csr(&this->mklSparseMatrix, SPARSE_INDEX_BASE_ZERO, this->size, this->size, this->pointerB.data(),
                            this->pointerE.data(), this->colIndex.data(), this->nzValues.data());
}


sparse_matrix_t csr_matrix::getMKLSparseMatrix() const{
    return this->mklSparseMatrix;
}

sparse_matrix_t * csr_matrix::getMKLSparseMatrixPointer(){
    return &this->mklSparseMatrix;
}

matrix_descr csr_matrix::getMKLDescription() const {
    return this->mklDescription;
}


//insert row in the csr matrix, used when initializing the matrix
void csr_matrix::insertRow(vector<SparseTriplet> row, int rowId) {
    sort(row.begin(), row.end());
    for (auto & i : row) {
        if (i.value != 0) {
            this->nzValues.push_back(i.value);
            this->colIndex.push_back(i.col);
            this->nz++;
        }
    }
    this->rowPtr[rowId + 1] = this->nz;
}

//get values from a single row
vector<SparseTriplet> csr_matrix::getRow(int row) {
    vector<SparseTriplet> rowValues = vector<SparseTriplet>();
    long long int start = this->rowPtr[row];
    long long int end = this->rowPtr[row + 1];
    for (long long int i = start; i < end; i++) {
        SparseTriplet triplet(row, this->colIndex[i], this->nzValues[i]);
        rowValues.push_back(triplet);
    }
    return rowValues;
}

void csr_matrix::convertInternal(int sizel) {
    this->colIndex = vector<long long int>(sizel);
    this->pointerB = vector<long long int>(sizel);
    this->pointerE = vector<long long int>(sizel);
    int status = mkl_sparse_d_export_csr(this->mklSparseMatrix, (sparse_index_base_t *) SPARSE_INDEX_BASE_ZERO,
                            reinterpret_cast<long long int *>(&this->size),
                            reinterpret_cast<long long int *>(&this->size), reinterpret_cast<long long int **>(this->pointerB.data()),
                            reinterpret_cast<long long int **>(this->pointerE.data()),
                            reinterpret_cast<long long int **>(this->colIndex.data()),
                            reinterpret_cast<double **>(this->nzValues.data()));

    cout << "status: " << status << endl;
}

void csr_matrix::printAttr() const{
    cout << "size: " << this->size << endl;
    cout << "nz: " << this->nz << endl;
    cout << "nzValues: ";
    for (double nzValue : this->nzValues) {
        cout << nzValue << " ";
    }
    cout << endl;

    cout << "colIndex: ";
    for (long long i : this->colIndex) {
        cout << i << " ";
    }
    cout << endl;
    cout << "rowPtr: ";
    for (long long i : this->rowPtr) {
        cout << i << " ";
    }
    cout << endl;

    cout << "pointerB: ";
    for (long long i : this->pointerB) {
        cout << i << " ";
    }
    cout << endl;

    cout << "pointerE: ";
    for (long long i : this->pointerE) {
        cout << i << " ";
    }
    cout << endl;
}

double csr_matrix::getValue(int row, int col) const {
    long long int start = this->rowPtr[row];
    long long int end = this->rowPtr[row + 1];
    for (long long int i = start; i < end; i++) {
        if (this->colIndex[i] == col) {
            return this->nzValues[i];
        }
    }
    return 0;
}

int * csr_matrix::getRowPtr() const {
    return (int *) this->rowPtr.data();
}

void csr_matrix::saveMatrixMarketFile(string & filename) {
    ofstream file(filename);
    cout << "Saving matrix to file: " << filename << endl;
    file << "%%MatrixMarket matrix coordinate real general" << endl;
    file << this->size << " " << this->size << " " << this->nz << endl;
    for (int i = 0; i < this->size; i++) {
        for (int j = (int) this->rowPtr[i]; j < this->rowPtr[i + 1]; j++) {
            file << i + 1 << " " << this->colIndex[j] + 1 << " " << this->nzValues[j] << endl;
        }
    }
    file.close();
}
