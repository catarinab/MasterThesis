#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <bits/stdc++.h>

#include "../utils/scalar_ops.cpp" 
#include "../utils/mtx_ops.cpp"

using namespace std;
#define epsilon 1e-12 //10^-12
#define omega 0.0001

bool debugMtr = false;
bool vecFile = false;
int beta = 1;


int arnoldiIteration(CSR_Matrix A, Vector b, int n, int m, int me, int nprocs, dense_Matrix * V, dense_Matrix * H) {
    int helpSize = 0;
    int sendEnd = ENDTAG;

    b = b / b.getNorm2();
    V->setCol(0, b);

    int k = 1;

    //auxiliar
    Vector opResult(m);

    for(k = 1; k < n + 1; k++) {

        Vector w = sparseMatrixVector(A, V->getCol(k-1), 0, m, m);


        for(int j = 0; j <= k; j++) {
            H->setValue(j, k-1, dotProduct(w, V->getCol(j), 0, m));
            opResult = V->getCol(j) * H->getValue(j, k-1);
            w = subtractVec(w, opResult, 0, m);
        }


        H->setValue(k, k - 1, w.getNorm2());
        if(H->getValue(k, k - 1) > epsilon)
            V->setCol(k, w / H->getValue(k, k - 1));

        else
            return k;
    }
    return --k;
}



int findM(dense_Matrix H) {
    int m = 1; //2^0
    double norm = H.getNorm2();

    while(norm / m >= 1) {
        m *= 2;
    }

    return m;
}

double pade_coeff(int p, int q, int j ) {
    return (factorial(p)*factorial(p+q-j))/(factorial(p+q)*factorial(p-j)*factorial(j));
}


dense_Matrix pade_terms(int p, int q, dense_Matrix A) {
    //initialization
    dense_Matrix res(A.getRowVal(), A.getColVal());
    dense_Matrix prevPower(A.getRowVal(), A.getColVal());
    prevPower.setIdentity();
    res = denseMatrixMatrixAdd(res, prevPower * pade_coeff(p, q, 0));

    for(int j = 1; j <= p; j++) {
        double coeff = pade_coeff(p, q, j);
        prevPower = denseMatrixMatrixMult(prevPower, A);
        res = denseMatrixMatrixAdd(res, prevPower * coeff);
    }
    return res;
}


dense_Matrix padeApproximation(dense_Matrix A, int m, int k) {
    return denseMatrixMatrixDiv(pade_terms(k, m, A), pade_terms(m, k, -A));
}


int main (int argc, char* argv[]) {
    double exec_time;
    bool vecFile = false;

    int krylovDegree = 3;
    int finalKrylovDegree;

    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx("/home/cat/uni/thesis/project/mtx/Trefethen_20b/Trefethen_20b.mtx");
    int size = csr.getSize();
    Vector b(size);
    b.getOnesVec();


    dense_Matrix V(size, krylovDegree + 1);
    dense_Matrix H(krylovDegree , krylovDegree);
    exec_time = -omp_get_wtime();
    //from this, we get the Orthonormal basis of the Krylov subspace (V) and the upper Hessenberg matrix (H)
    finalKrylovDegree = arnoldiIteration(csr, b, krylovDegree, size, 0, 1, &V, &H);

    V.printAttr("V");
    H.printAttr("H");
        
    V.printAttr("V");
    H.printAttr("H");
    cout << "finalKrylovDegree: " << finalKrylovDegree << endl;
    
    //m is the smallest power of two such that ||A||/m <= 1
    int m = findM(H);

    cout << "m: " << m << endl;

    dense_Matrix scaledH = H/m;
    scaledH.printAttr("scaledH");


    //como escolher o p e o q ?????
    int p = m;
    int q = m;
    dense_Matrix scaledExpH = padeApproximation(scaledH, p, q);

    scaledExpH.printAttr("scaledExpH");

    // a seguir, dar unscale

    exec_time += omp_get_wtime();
    cout << "exec_time: " << exec_time << endl;
    

    return 0;
}