#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix() {
    vector<vector<double>> A{
        {10, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {1, 20, 1, 2, 3, 4, 5, 6, 7, 8},
        {2, 1, 30, 1, 2, 3, 4, 5, 6, 7},
        {3, 2, 1, 40, 1, 2, 3, 4, 5, 6},
        {4, 3, 2, 1, 50, 1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1, 60, 1, 2, 3, 4},
        {6, 5, 4, 3, 2, 1, 70, 1, 2, 3},
        {7, 6, 5, 4, 3, 2, 1, 80, 1, 2},
        {8, 7, 6, 5, 4, 3, 2, 1, 90, 1},
        {9, 8, 7, 6, 5, 4, 3, 2, 1, 100}
    };
    return A;
}


//construir tendo em conta as matrizes do matlab?
vector<double> buildVector() {
    vector<double> b{2, 8, 9, 2, 3, 4, 5, 6, 7, 8};
    return b;
}


vector<double> subtractVec(vector<double> a, vector<double> b, int begin, int end) {
    vector<double> res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res[resIndex] = a[i] - b[i];
    }
    return res;
}

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int begin, int end, int size) {
    vector <double> res(end - begin);
    int resIndex = 0;

	for (int i = begin; i < end; i++) {
        resIndex = i - begin;
		res[resIndex] = 0;
		for (int j = 0; j < size; j++) {
			res[resIndex] += matrix[i][j] * v[j];
		}
	}
	return res;
}

double dotProduct(vector<double> a, vector<double> b, int begin, int end) {
	double dotProd = 0.0;

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a[i] * b[i]);
	}
    return dotProd;
}