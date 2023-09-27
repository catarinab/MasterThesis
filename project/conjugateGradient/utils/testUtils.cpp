#include "read_file_mtx.cpp"
#include "csr_matrix.cpp"

int rows, cols, nz;
CSR_Matrix * csrMatrix;

int main() {
    readFile("../mtx/LFAT5/LFAT5.mtx", &rows, &cols, &nz);
    csrMatrix = new CSR_Matrix(cols);
    cout << "vamos tentar enviar as rows: " << endl;
    for (int i = 0; i < rows; i++) {
        csrMatrix->insertRow(rowValues[i], i);
    }
    vector<SparseTriplet> row1 = csrMatrix->getRow(1);
    for (int i = 0; i < row1.size(); i++) {
        cout << "col: " << row1[i].col << " ";
        cout << row1[i].value << " ";
    }
    cout << endl;
    csrMatrix->printAttr();
    return 0;
}