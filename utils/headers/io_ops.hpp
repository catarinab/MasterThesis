/*File to read Matrix Market ?*/
#ifndef IO_OPS_HPP_HPP
#define IO_OPS_HPP_HPP

#include <string>
#include <vector>
#include "sparse_structs.h"
using namespace std;

vector<vector<SparseTriplet>> readFile_mtx(string inputFile, int * rows, int * cols, int * nz);
vector<double> readFile_vec(string inputFile, int size);

#endif // IO_OPS_HPP_HPP