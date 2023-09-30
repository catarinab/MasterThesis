
struct SparseTriplet {
    int col;
    int row;
    double value;
    SparseTriplet(int row, int col, double value) : row(row), col(col), value(value) {}
    SparseTriplet() : row(0), col(0), value(0) {}
};

bool operator<( const SparseTriplet& a, const SparseTriplet&b ){
    return a.col < b.col;
}

struct SparseDouble {
    int col;
    double value;

    SparseDouble(int col, double value) : col(col), value(value) {}
    SparseDouble() : col(0), value(0) {}
};

bool operator<( const SparseDouble& a, const SparseDouble&b ){
    return a.col < b.col;
}
