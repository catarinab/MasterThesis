#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "../utils/distr_mtx_ops.cpp"
#include "../utils/helpProccess.cpp"
#include "../utils/scalar_ops.cpp"

int findM(double norm, int theta , int * power) {
    int m = 1; //2^0
    *power = 0; //2^0

    while(norm / m >= theta) {
        m *= 2;
        (*power)++;
    }

    return m;
}


vector<double> get_pade_coefficients(int m) {
    vector<double> coeff;
    if(m == 3)
        coeff = {120, 60, 12, 1};
    else if(m == 5)
        coeff = {30240, 15120, 3360, 420, 30, 1};
    else if(m == 7)
        coeff = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
    else if (m == 9)
        coeff = {17643225600, 8821612800, 2075673600, 302702400, 30270240,
                2162160, 110880, 3960, 90, 1};
    else if (m == 13)
        coeff = {64764752532480000, 32382376266240000, 7771770303897600,
                1187353796428800,  129060195264000,   10559470521600,
                670442572800,      33522128640,       1323241920,
                40840800,          960960,            16380,  182,  1};
    return coeff;
}


int definePadeParams(vector<dense_Matrix> * powers, int * s, int * power, dense_Matrix A) {
    vector<double> theta = {
    1.495585217958292e-002, // m_vals = 3
    2.539398330063230e-001,  // m_vals = 5
    9.504178996162932e-001,  // m_vals = 7
    2.097847961257068e+000,  // m_vals = 9
    5.371920351148152e+000}; // m_vals = 13


    dense_Matrix identity = dense_Matrix(A.getColVal(), A.getRowVal());
    identity.setIdentity();

    double normA = A.getNorm2();

    cout << "normA: " << normA << endl;

    vector<dense_Matrix>::iterator ptr = powers->begin();

    *ptr++ = identity;
    *ptr++ = A;
    dense_Matrix A2 = denseMatrixMatrixMult(A, A);
    *ptr++ = A2;
    dense_Matrix A4 = denseMatrixMatrixMult(A2, A2);
    *++ptr = A4;
    dense_Matrix A6 = denseMatrixMatrixMult(A2, A4);
    *(ptr + 2) = A6;


    if(normA < theta[0])
        return 3;
    else if(normA < theta[1])
        return 5;


    dense_Matrix A8 = denseMatrixMatrixMult(A4, A4);

    powers->push_back(A8);

    if(normA < theta[2])
        return 7;
    if(normA < theta[3])
        return 9;
    if(normA < theta[4])
        return 13;

    //so neste caso fazemos o scaling!
    *s = findM(normA, theta[4], power);
    return 13;

}