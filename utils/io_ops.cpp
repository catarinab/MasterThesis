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

long long int getMtxSize(const string& inputFile) {
    ifstream file(inputFile);
    string line;
    
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        for(double & value : values) {
            getline(ss, line, ' ');
            value = stod(line);
        }
        break;
    } 

    file.close();
    return (long long int) values[0];
}

vector<vector<SparseTriplet>> readFile_full_mtx(const string& inputFile, long long int * rows, long long int * cols, long long int * nz) {
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        if(!isDefined) {
            for(double & value : values) {
                getline(ss, line, ' ');
                value = stod(line);
            }
            *rows = (long long int) values[0];
            *cols = (long long int) values[1];
            *nz = (long long int) values[2];
            rowValues.resize(*rows);
            isDefined = true;
        }
        else {
            for(double & value : values) {
                getline(ss, line, ' ');
                value = stod(line);
            }
            rowValues[(long long int) values[0] - 1]
                .emplace_back( (long long int) values[0] - 1, (long long int) values[1] - 1, values[2]);
        }
    } 

    file.close();
    return rowValues;
}


vector<vector<SparseTriplet>> readFile_part_mtx(const string& inputFile, long long int * rows, long long int * cols, long long int * nz,
                                                int * displs, int * counts, int me) {
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        stringstream ss(line);
        if(!isDefined) {
            for(double & value : values) {
                getline(ss, line, ' ');
                value = stod(line);
            }
            *rows = (long long int) values[0];
            *cols = (long long int) values[1];
            *nz = (long long int) values[2];
            rowValues.resize(counts[me]);
            isDefined = true;
        }
        else {
            for(double & value : values) {
                getline(ss, line, ' ');
                value = stod(line);
            }
            if((int) values[0] - 1 >= displs[me] && (long long int) values[0] - 1 < displs[me] + counts[me]) {
                rowValues[(long long int) values[0] - 1 - displs[me]]
                        .emplace_back( (long long int) values[0] - 1, (long long int) values[1] - 1, values[2]);
            }
               
        }
    } 
    
    file.close();
    return rowValues;
}