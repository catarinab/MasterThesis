#ifndef IO_OPS_HPP
#define IO_OPS_HPP

#include <string>
#include <vector>
#include "utils.hpp"
using namespace std;

vector<vector<SparseTriplet>> readFile_mtx(string inputFile, int * rows, int * cols, int * nz);

#endif // IO_OPS_HPP