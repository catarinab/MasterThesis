#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

#include "headers/io_ops.hpp"

/*
    Read Matrix Market file and return a vector of SparseTriplets
    Each SparseTriplet is a row of the matrix
*/

using namespace std;

static constexpr int io_buffer_size = 1 << 20;

pair<double, double> readHeader(string& inputFile) {
    double values[3];
    string line;
    ifstream file(inputFile);

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
    return make_pair(values[0], values[2]);

}

pair<double, double> readHeader(ifstream& file) {
    double values[3];
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
    return make_pair(values[0], values[2]);
}

vector<vector<SparseTriplet>> readFile_full_mtx(const string& inputFile, long long int * rows, long long int * cols,
                                                long long int * nz) {
    ifstream file(inputFile);
    vector<vector<SparseTriplet>> rowValues;
    pair<double, double> header = readHeader(file);
    *rows = (long long int) header.first;
    *cols = (long long int) header.first;
    *nz = (long long int) header.second;
    rowValues.resize(*rows);

    string line;

    std::vector<char> input(io_buffer_size, '\0');

    while (file)
    {
        char* ptr = input.data();

        file.read(ptr, io_buffer_size);

        if (file.gcount() > 0)
        {
            input.resize(file.gcount());

            // Find the end of the last complete line
            char* last = file ? std::strrchr(ptr, '\n') : &input.back();
            int backtrace = last - &input.back();

            // If the last line is truncated, rewind file pointer
            // to the beginning of this line, so it can be read again
            // in the next block.
            if (backtrace < 0)
            {
                file.seekg(backtrace, file.cur);
                std::fill(last + 1, &input.back(), '\0');
            }

            // Parse each line and then add to the array;
            while (ptr < last)
            {

                long row = strtol(ptr, &ptr, 10) - 1;
                long col = strtol(ptr, &ptr, 10) - 1;
                double value = strtod(ptr, &ptr);

                ++ptr;

                SparseTriplet triplet(row, col, value);

                rowValues[row].push_back(triplet);
            }
        }
    }
    file.close();
    return rowValues;
}


vector<vector<SparseTriplet>> readFile_part_mtx(const string& inputFile, long long int * rows, long long int * cols, long long int * nz,
                                                const int * displs, const int * counts, int me) {
    ifstream file(inputFile);
    vector<vector<SparseTriplet>> rowValues;
    pair<double, double> header = readHeader(file);
    *rows = (long long int) header.first;
    *cols = (long long int) header.first;
    *nz = (long long int) header.second;
    rowValues.resize(counts[me]);

    string line;

    std::vector<char> input(io_buffer_size, '\0');

    while (file)
    {
        char* ptr = input.data();

        file.read(ptr, io_buffer_size);

        if (file.gcount() > 0)
        {
            input.resize(file.gcount());

            // Find the end of the last complete line
            char* last = file ? std::strrchr(ptr, '\n') : &input.back();
            int backtrace = last - &input.back();

            // If the last line is truncated, rewind file pointer
            // to the beginning of this line, so it can be read again
            // in the next block.
            if (backtrace < 0)
            {
                file.seekg(backtrace, file.cur);
                std::fill(last + 1, &input.back(), '\0');
            }

            // Parse each line and then add to the array;
            while (ptr < last)
            {
                long row = strtol(ptr, &ptr, 10) - 1;
                long col = strtol(ptr, &ptr, 10) - 1;
                double value = strtod(ptr, &ptr);

                if((int) row >= displs[me] && (int) row < displs[me] + counts[me]) {
                    SparseTriplet triplet(row, col, value);

                    rowValues[row - displs[me]].push_back(triplet);
                }

                ++ptr;
            }
        }
    }
    file.close();
    return rowValues;
}