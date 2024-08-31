#ifndef DENSE_VECTOR_HPP
#define DENSE_VECTOR_HPP

#include <utility>
#include <vector>
#include <ostream>

class dense_vector {
public:
    int size; // Number of columns
    std::vector<double> values;

    inline dense_vector() : size(0), values(std::vector<double>(0)) {}
    inline explicit dense_vector(int size) : size(size), values(std::vector<double>(size)) {}
    dense_vector(int size, std::vector<double> values) : size(size) {
        this->values = std::move(values);
    }

    void getOnesVec();
    void insertValue(int col, double value);
    void setValues(std::vector<double> values);
    void setValue(int i, double value);
    
    dense_vector operator*(double x);
    dense_vector operator/(double x);
    dense_vector operator/=(double x);
    dense_vector operator-(const dense_vector& other) const;

    // Member functions
    double getNorm2();
    friend std::ostream& operator<<(std::ostream& os, const dense_vector& dv);
};

#endif // DENSE_VECTOR_HPP
