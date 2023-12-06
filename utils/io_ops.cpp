/*File to read Matrix Market ?*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

double values[3];

vector<vector<SparseTriplet>> rowValues;

void initRows(int rows, int cols) {
    rowValues.resize(rows);
}

void processLine(string line) {
    stringstream ss(line);
    for(int i = 0; i < 3; i++) {
        getline(ss, line, ' ');
        values[i] = stod(line);
    }
    rowValues[(int) values[0] - 1]
            .push_back(SparseTriplet( (int) values[0] - 1, (int) values[1] - 1, values[2]));
}

vector<vector<SparseTriplet>> readFile_mtx(string inputFile, int * rows, int * cols, int * nz) {
    //paralelizar ?
    
    ifstream file(inputFile);
    string line;
    bool isDefined = false;
    while (getline(file, line)) {
        if(line[0] == '%') continue;
        if(!isDefined) {
            stringstream ss(line);
            for(int i = 0; i < 3; i++) {
                getline(ss, line, ' ');
                values[i] = stod(line);
            }
            *rows = values[0];
            *cols = values[1];
            *nz = values[2];
            initRows(values[0], values[1]);
            isDefined = true;
        }
        else {
            #pragma omp task firstprivate(line)
                processLine(line);
        }
    } 
        
    #pragma omp taskwait
    file.close();

    return rowValues;
}

vector<double> readFile_vec(string inputFile, int size) {
    //paralelizar ?
    ifstream file(inputFile);
    string line;
    vector<double> vec(size);
    int counter = 0;
    bool isDefined = false;

    while (getline(file, line)) {
        if(line[0] == '%') continue;
        vec[counter++] = stod(line);
    }
    file.close();

    return vec;
}