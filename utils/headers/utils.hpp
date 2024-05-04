#ifndef UTILS_HPP
#define UTILS_HPP

#define ROOT 0

#define ENDTAG (-1)

#define MV 3
#define VV 4
#define ADD 6

#define EPS16 2.220446049250313e-16

#define PI 3.141592653589793238466

#include <complex>

#include "dense_matrix.hpp"
#include "dense_vector.hpp"

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

inline double factorial(int n) {
    double result = 1;
    for(int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

inline double falling_factorial(double n, int k) {
    double result = 1;
    for(int i = 0; i < k ; i++) {
        result *= (n - i);
    }
    return result;
}


#endif // UTILS_HPP