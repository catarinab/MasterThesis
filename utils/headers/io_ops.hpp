#ifndef IO_OPS_HPP
#define IO_OPS_HPP

#include <string>
#include <vector>
#include "utils.hpp"
using namespace std;

int getMtxSize(string inputFile);
vector<vector<SparseTriplet>> readFile_full_mtx(string inputFile, int * rows, int * cols, int * nz);
vector<vector<SparseTriplet>> readFile_part_mtx(string inputFile, int * rows, int * cols, int * nz
                                                , int * displs, int * counts, int me);
#endif // IO_OPS_HPP