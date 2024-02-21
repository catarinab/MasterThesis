#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#define ROOT 0

#define ENDTAG -1

#define MV 3
#define VV 4
#define SUB 5
#define ADD 6

#define EPS 2e-52
#define EPS16 2e-16


struct SparseTriplet {
    long long int col;
    long long int row;
    double value;
    SparseTriplet(long long int row, long long int col, double value) : row(row), col(col), value(value) {}
    SparseTriplet() : row(0), col(0), value(0) {}
};


inline bool operator<(const SparseTriplet& a, const SparseTriplet& b) {
    if(a.row < b.row || (a.row == b.row && a.col < b.col))
        return true;
    return false;
}

#endif // STRUCTS_HPP