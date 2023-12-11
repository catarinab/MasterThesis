#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#define ROOT 0

#define ENDTAG -1

#define MV 3
#define VV 4
#define SUB 5
#define ADD 6

struct SparseTriplet {
    int col;
    int row;
    double value;
    SparseTriplet(int row, int col, double value) : row(row), col(col), value(value) {}
    SparseTriplet() : row(0), col(0), value(0) {}
};

inline bool operator<(const SparseTriplet& a, const SparseTriplet& b) {
    if(a.row < b.row)
        return true;
    else if(a.row == b.row && a.col < b.col)
        return true;
    else
        return false;
}

#endif // STRUCTS_HPP