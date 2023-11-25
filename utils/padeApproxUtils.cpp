#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "../utils/distr_mtx_ops.cpp"
#include "../utils/helpProccess.cpp"

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


dense_Matrix definePadeParams(vector<dense_Matrix> * powers, int * m, int * power, dense_Matrix A) {
    vector<double> theta = {
    1.495585217958292e-002, // m_val = 3
    2.539398330063230e-001,  // m_val = 5
    9.504178996162932e-001,  // m_val = 7
    2.097847961257068e+000,  // m_val = 9
    5.371920351148152e+000}; // m_val = 13

    dense_Matrix resultingMatrix = A;

    *m = 0;

    dense_Matrix identity = dense_Matrix(A.getColVal(), A.getRowVal());
    identity.setIdentity();

    double normA = A.getNorm2();

    if(normA < theta[0])
        *m = 3;
    else if(normA < theta[1])
        *m = 5;
    else if(normA < theta[2])
        *m = 7;
    else if(normA < theta[3])
        *m = 9;
    else if(normA < theta[4])
        *m = 13;

    if(*m == 0) {
        int s = findM(normA, theta[4], power);
        *m = 13;
        resultingMatrix = A/(s);
    }
    

    vector<dense_Matrix>::iterator ptr = powers->begin();

    *ptr++ = identity;
    *ptr++ = resultingMatrix;
    dense_Matrix res2 = denseMatrixMult(resultingMatrix, resultingMatrix);
    *ptr++ = res2;
    dense_Matrix res4 = denseMatrixMult(res2, res2);
    *++ptr = res4;
    dense_Matrix res6 = denseMatrixMult(res2, res4);
    *(ptr + 2) = res6;
    dense_Matrix res8 = denseMatrixMult(res4, res4);
    powers->push_back(res8);

    return resultingMatrix;

}