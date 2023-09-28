/*File to read Matrix Market ?*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#ifndef SPARSE_TRIPLET_H
#define SPARSE_TRIPLET_H 1
    #include "sparse_triplet.h"
#endif

using namespace std;

double values[3];

vector<vector<SparseTriplet>> rowValues;

void initRows(int rows, int cols) {
    rowValues.resize(rows);
}

vector<vector<SparseTriplet>> readFile_mtx(string inputFile, int * rows, int * cols, int * nz) {
    //paralelizar ?
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        for(int i = 0; i < 3; i++) {
            getline(ss, line, ' ');
            values[i] = stod(line);
        }
        if(!isDefined) {
            *rows = values[0];
            *cols = values[1];
            *nz = values[2];
            initRows(values[0], values[1]);
            isDefined = true;
        } else
            rowValues[(int) values[0] - 1]
                .push_back(SparseTriplet( (int) values[0] - 1, (int) values[1] - 1, values[2]));
        
    }
    file.close();

    return rowValues;
}