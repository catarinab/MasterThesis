#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "headers/io_ops.hpp" 

/*
    Read Matrix Market file and return a vector of SparseTriplets
    Each SparseTriplet is a row of the matrix
*/

using namespace std;

double values[3];

vector<vector<SparseTriplet>> rowValues;

int getMtxSize(string inputFile) {
    ifstream file(inputFile);
    string line;
    
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        for(int i = 0; i < 3; i++) {
            getline(ss, line, ' ');
            values[i] = stod(line);
        }
        break;
    } 

    file.close();
    return values[0];
}

vector<vector<SparseTriplet>> readFile_full_mtx(string inputFile, int * rows, int * cols, int * nz) {
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        if(!isDefined) {
            for(int i = 0; i < 3; i++) {
                getline(ss, line, ' ');
                values[i] = stod(line);
            }
            *rows = values[0];
            *cols = values[1];
            *nz = values[2];
            rowValues.resize(*rows);
            isDefined = true;
        }
        else {
            for(int i = 0; i < 3; i++) {
                getline(ss, line, ' ');
                values[i] = stod(line);
            }
            rowValues[(int) values[0] - 1]
                .push_back(SparseTriplet( (int) values[0] - 1, (int) values[1] - 1, values[2]));
        }
    } 

    file.close();
    return rowValues;
}


vector<vector<SparseTriplet>> readFile_part_mtx(string inputFile, int * rows, int * cols, int * nz, int * displs, int * counts, int me) {
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    int startRow = 100000;
    int endRow = -1;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        if(!isDefined) {
            for(int i = 0; i < 3; i++) {
                getline(ss, line, ' ');
                values[i] = stod(line);
            }
            *rows = values[0];
            *cols = values[1];
            *nz = values[2];
            rowValues.resize(counts[me]);
            isDefined = true;
        }
        else {
            for(int i = 0; i < 3; i++) {
                getline(ss, line, ' ');
                values[i] = stod(line);
            }
            if((int) values[0] - 1 >= displs[me] && (int) values[0] - 1 < displs[me] + counts[me]) {
                rowValues[(int) values[0] - 1 - displs[me]]
                        .push_back(SparseTriplet( (int) values[0] - 1, (int) values[1] - 1, values[2]));
            }
               
        }
    } 
    
    file.close();
    return rowValues;
}