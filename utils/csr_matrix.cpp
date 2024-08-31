#include "headers/csr_matrix.hpp"
#include <fstream>

using namespace std;


long long int csr_matrix::getSize() const {
    return this->size;
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
