#ifndef DENSE_VECTOR_HPP
#define DENSE_VECTOR_HPP

#include <vector>

class dense_vector {
public:
    int size; // Number of columns
    std::vector<double> values;

    inline dense_vector() : size(0), values(std::vector<double>(0)) {}
    inline dense_vector(int size) : size(size), values(std::vector<double>(size)) {}

    int getSize();

    void setValues(std::vector<double> values);
    void resize(int size);
    void getRandomVec();
    void getOnesVec();
    void insertValue(int col, double value);

    dense_vector operator*(double x);
    dense_vector operator/(double x);

    // Member functions
    double getNorm2();
    void printAttr(std::string name);
};

#endif // DENSE_VECTOR_HPP
