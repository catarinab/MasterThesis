#ifndef DENSE_VECTOR_HPP
#define DENSE_VECTOR_HPP

#include <utility>
#include <vector>

class dense_vector {
public:
    int size; // Number of columns
    std::vector<double> values;

    inline dense_vector() : size(0), values(std::vector<double>(0)) {}
    inline explicit dense_vector(int size) : size(size), values(std::vector<double>(size)) {}
    dense_vector(int size, std::vector<double> values) : size(size) {
        this->values = std::move(values);
    }

    [[nodiscard]] int getSize() const;
    
    void resize(int size);
    void getOnesVec();
    void getMaxValVec();
    void insertValue(int col, double value);
    void setValues(std::vector<double> values);
    void setValue(int i, double value);
    
    double getValue(int i);
    [[nodiscard]] std::vector<double> getValues() const;
    
    dense_vector operator*(double x);
    dense_vector operator/(double x);

    // Member functions
    double getNorm2();

    void getZeroVec();
};

#endif // DENSE_VECTOR_HPP
