#include <iostream>
#include <mkl.h>
#include <algorithm>
#include <numeric> // iota
#include <complex>
#include <cmath>

#include "../utils/headers/dense_matrix.hpp"
#include "../utils/headers/schur-blocking.hpp"

/* 
Algorithm based on paper "A Schur–Parlett Algorithm for Computing Matrix Functions"
by Davies, Philip I. and Higham, Nicholas J.
*/

using namespace std;

double delta = 0.1;

void getSubMatrix(double ** subMatrix, const double * matrix, int row, int subSize, int size) {
    for(int i = row; i < row + subSize; i++) {
        for(int j = row; j < row + subSize; j++) {
            (*subMatrix)[(i - row) * subSize + (j - row)] = matrix[i * size + j];
        }
    }
}

vector<int> findIndices(const vector<int>& q, int val) {
    vector<int> indices = vector<int>();
    for (int i = 0; i < q.size(); ++i) {
        if (q[i] == val)
            indices.push_back(i);
    }
    return indices;
}

vector<int> sortIndices(const vector<double>& g) {
    vector<int> indices(g.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&g](int a, int b) { return g[a] < g[b]; });
    return indices;
}

void rsf2csf(double * T, double * U, complex<double> ** T_csf, complex<double> ** U_csf, int size, complex<double> ** w) {
    //copy double values to complex values
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            (*T_csf)[i * size + j] = complex<double>(T[i * size + j], 0);
            (*U_csf)[i * size + j] = complex<double>(U[i * size + j], 0);
        }
    }

    //Row major
    auto * G = (complex<double> *) calloc(2 * 2, sizeof(complex<double>));
    auto * subMatrix = (double *) calloc(2 * 2, sizeof(double));
    vector<complex<double>> mu = vector<complex<double>>(2);
    complex<double> temp1;
    complex<double> temp2;

    double wr[2], wi[2]; //eigenvalues

    for(int m = size - 1; m >= 1; m--) {
        if(T[m * size + (m-1)] != 0) {
            int lda = 2;
            //T(m-1:m, m-1:m)
            getSubMatrix(&subMatrix, T, m - 1, lda, size);

            int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', lda, subMatrix, lda,
                          wr, wi, nullptr, lda, nullptr, lda);
            if(info != 0)
                cout << "LAPACKE_dgeev info: " << info << endl;

            //mu = eig(T(k,k)) - T(m,m); -> T(k,k) = T(m-1:m, m-1:m)
            mu[0] = complex<double>(wr[0], wi[0]) - T[m * size + m];
            mu[1] = complex<double>(wr[1], wi[1]) - T[m * size + m];

            // r = hypot(mu(1), T(m,m-1));
            double r = sqrt(pow(mu[0].real() - T[m * size + (m-1)], 2) + pow(mu[0].imag() , 2));
            //c = mu(1)/r;
            complex<double> c = mu[0] / r;
            //s = T(m,m-1)/r;
            double s = T[m * size + (m-1)] / r;

            //G = [c' s; -s c];
            G[0] = conj(c);
            G[1] = s;
            G[2] = -s;
            G[3] = c;

            //T_csf(k,m-1:n) = G*T(k,m-1:n);
            for(int j = m - 1; j < size; j++) {
                temp1 = G[0] * (*T_csf)[(m-1) * size + j] + G[1] * (*T_csf)[m * size + j];
                temp2 = G[2] * (*T_csf)[(m-1) * size + j] + G[3] * (*T_csf)[m * size + j];
                (*T_csf)[(m-1) * size + j] = temp1;
                (*T_csf)[m * size + j] = temp2;
            }

            //(*T_csf)(1:m,k) = T(1:m,k)*G';
            for(int j = 0; j <= m; j++) {
                temp1 = (*T_csf)[j * size + m - 1] * conj(G[0]) + (*T_csf)[j * size + m] * conj(G[1]);
                temp2 = (*T_csf)[j * size + m - 1] * conj(G[2]) + (*T_csf)[j * size + m] * conj(G[3]);
                (*T_csf)[j * size + m - 1] = temp1;
                (*T_csf)[j * size + m] = temp2;
            }
            //(*U_csf)(:,k) = U(:,k)*G';
            for(int j = 0; j < size; j++) {
                temp1 = (*U_csf)[j * size + m - 1] * conj(G[0]) + (*U_csf)[j * size + m] * conj(G[1]);
                temp2 = (*U_csf)[j * size + m - 1] * conj(G[2]) + (*U_csf)[j * size + m] * conj(G[3]);
                (*U_csf)[j * size + m - 1] = temp1;
                (*U_csf)[j * size + m] = temp2;
            }
            (*T_csf)[m * size + (m-1)] = 0;
        }
    }

    //create diagonal vector
    for(int i = 0; i < size; i++)
        (*w)[i] = (*T_csf)[i * size + i];

    free(subMatrix);
    free(G);

}

vector<vector<int>> swapping(vector<int>& q, vector<int> * ILST, vector<int> * IFST) {
    int mMax = *max_element(q.begin(), q.end());
    vector<double> g(mMax + 1, 0);
    vector<int> phi(mMax + 1, 0);

    vector<vector<int>> ind = vector<vector<int>>();

    for (int i = 0; i <= mMax; ++i) {
        vector<int> p = findIndices(q, i);
        phi[i] = (int) p.size();
        for(int val: p)
            g[i] += val + 1;
        g[i] /= (double) phi[i];
    }

    vector<int> y = sortIndices(g);

    int beta = 0;
    for (int i : y) {
        if (any_of(q.begin() + beta, q.begin() + beta + phi[i], [&](int val) { return val != i ;})) {
            //f = find(q == i)
            vector<int> f = findIndices(q, i);

            //g = beta : beta + h[i] - 1
            vector<int> g_indices = vector<int>();
            for (int ii = beta; ii < beta + phi[i]; ii++) {
                g_indices.push_back(ii);
            }


            vector<int> fValNotG = vector<int>();
            vector<int> gValNotF = vector<int>();

            //f(f!=g) = fValNotG
            //g(f!=g) = gValNotF
            for(int j = 0; j < f.size(); j++){
                if(f[j] != g_indices[j]){
                    fValNotG.push_back(f[j]);
                    gValNotF.push_back(g_indices[j]);
                }
            }

            //Concatenate gValNotF to ILST and fValNotG to IFST
            for(int val : gValNotF)
                ILST->push_back(val);
            for(int val : fValNotG)
                IFST->push_back(val);


            //Let v = beta:f(end) and delete all elements of v that are elements of f.
            vector<int> v;
            for (int j = beta; j <= f.back(); j++) {
                if (find(f.begin(), f.end(), j) == f.end()){
                    v.push_back(j);
                }
            }

            //q(g(end) + 1: f(end)) = q(v)
            vector<int> oldQ = q;
            int vIdx = 0;
            for(int ii = g_indices.back() + 1; ii <= f.back(); ii++)
                q[ii] = oldQ[v[vIdx++]];


            //q[g] = [i,...,i]
            for(int ii = g_indices[0]; ii <= g_indices.back(); ii++)
                q[ii] = i;

            beta += phi[i];
        }
    }

    //create vector of block/element indices
    vector<int> current_group;
    current_group.push_back(0);
    for (int i = 1; i < q.size(); ++i) {
        if (q[i] != q[i - 1]) {
            ind.push_back(current_group);
            current_group.clear();
        }
        current_group.push_back(i);
    }
    ind.push_back(current_group);

    return ind;
}

vector<int> blocking(int size, complex<double> * diag) {

    vector<int> m = vector<int>(size, 0);

    int maxM = 0;
    for(int i = 0; i < size; i++) {
        //if the element has not been selected yet.
        if(m[i] == 0) {
            m[i] = ++maxM;
        }

        for(int j = i + 1; j < size; j++) {
            if(m[i] != m[j]){
                if(fabs(diag[i] - diag[j]) <= delta){
                    if(m[j] == 0) {
                        m[j] = m[i];
                    }
                    else {
                        int p = max(m[i], m[j]);
                        int q = min(m[i], m[j]);
                        for(int k = 0; k < size; k++) {
                            if(m[k] == p)
                                m[k] = q;
                        }
                        for(int k = 0; k < size; k++) {
                            if(m[k] > p)
                                m[k] = m[k] - 1;
                        }
                        maxM--;
                    }
                }
            }
        }
    }

    //return to 0 indexing
    for(int i = 0; i < size; i++) {
        m[i] = m[i] - 1;
    }
    return m;
}

vector<vector<int>> schurDecomposition(double * A, complex<double> ** T, complex<double> ** U, int size) {
    auto * U_real = (double *) malloc(size * size * sizeof(double));
    auto * wr = (double *) calloc(size, sizeof(double));
    auto * wi = (double *) calloc(size, sizeof(double));
    auto * w = (complex<double> *) calloc(size, sizeof(complex<double>));

    vector<int> mm;
    vector<vector<int>> ind;
    vector<int> ILST = vector<int>();
    vector<int> IFST = vector<int>();



    int info = LAPACKE_dhseqr(LAPACK_ROW_MAJOR, 'S', 'I', size, 1, size, A , size, wr, wi,
                             U_real, size);
    if(info != 0)
        cout << "LAPACKE_dhseqr info: " << info << endl;

    //real schur form to complex schur form
    rsf2csf(A, U_real, T, U, size, &w);

    /*printMatlabFileComplex("T-16-before.txt", size, *T);
    printMatlabFileComplex("U-16-before.txt", size, *U);*/

    vector<int> clusters = blocking(size, w);

    ind = swapping(clusters, &ILST, &IFST);

    for(int i = 0; i < ILST.size(); i++) {
        info = LAPACKE_ztrexc(LAPACK_ROW_MAJOR, 'V', size, reinterpret_cast<MKL_Complex16 *>(*T),
                              size, reinterpret_cast<MKL_Complex16 *>(*U), size, IFST[i] + 1, ILST[i] + 1);
        if(info != 0)
            cout << "LAPACKE_ztrexc info: " << info << endl;
    }

    *T = reinterpret_cast<complex<double> *>(*T);
    *U = reinterpret_cast<complex<double> *>(*U);

    free(wr);
    free(wi);
    free(w);
    free(U_real);
    return ind;
}