/*File to read Matrix Market ?*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

double values[3];
vector<vector<double>> mtx;

vector<vector<double>> readFile_mtx(string inputFile) {
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
            mtx.resize(values[0]);
            for(int i = 0; i < values[0]; i++) {
                mtx[i].resize(values[1]);
            }
            isDefined = true;
        } else
            mtx[values[0] - 1][values[1] - 1] = values[2];
    }
    file.close();

    return mtx;
}