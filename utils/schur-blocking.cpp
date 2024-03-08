#include <iostream>
#include <omp.h>
#include <cstring>
#include <complex>
#include <mkl.h>
#include <algorithm>
#include <numeric> // iota

#include "../utils/headers/dense_matrix.hpp"
#include "../utils/headers/calculate-MLF.hpp"
#include "../utils/headers/schur-blocking.hpp"
#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../utils/headers/utils.hpp"

/* 
Algorithm based on paper "A Schur–Parlett Algorithm for Computing Matrix Functions"
by Davies, Philip I. and Higham, Nicholas J.
*/

using namespace std;

vector<int> findIndices(const vector<int>& q, int val) {
    vector<int> indices = vector<int>();
    for (int i = 0; i < q.size(); ++i) {
        if (q[i] == val)
            indices.push_back(i);
    }
    return indices;
}

vector<int> sortIndices(const vector<int>& g) {
    vector<int> indices(g.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&g](int a, int b) { return g[a] < g[b]; });
    return indices;
}

vector<int> sortAndCumSum(const vector<int>& h, const vector<int>& y) {
    vector<int> sorted_h(h.size());
    //rearrange h based on the indices of y
    transform(y.begin(), y.end(), sorted_h.begin(), [&h](int idx) { return h[idx]; });
    vector<int> cumSum_h(sorted_h.size() + 1);
    // cumSum = [0, cumsum(h))];
    partial_sum(sorted_h.begin(), sorted_h.end(), cumSum_h.begin() + 1);
    return cumSum_h;
}

vector<vector<int>> swapping(vector<int>& m, vector<int> * ILST, vector<int> * IFST) {
    int mMax = *max_element(m.begin(), m.end());

    vector<int> g(mMax + 1, 0);
    vector<int> h(mMax + 1, 0);

    vector<int> blockList(mMax + 1, 0);

    for (int i = 0; i <= mMax; ++i) {
        vector<int> p = findIndices(m, i);
        h[i] = (int) p.size();
        g[i] = accumulate(p.begin(), p.end(), 0);
        g[i] /= h[i];
    }

    vector<int> y = sortIndices(g);

    int beta = 0;
    int skippedIdx = 0;
    for (int i : y) {
        if (any_of(m.begin() + beta, m.begin() + beta + h[i] -1, [&](int val) { return val != i; })) {
            vector<int> f = vector<int>();
            //f = find(q == i)
            for (int ii = 0; ii < m.size(); ii++) {
                if (m[ii] == i) {
                    f.push_back(ii);
                }

            }
            //g = beta : beta + h[i] - 1
            vector<int> g_indices = vector<int>();
            for (int ii = beta; ii < beta + h[i]; ii++) {
                g_indices.push_back(ii);
            }

            //Let v = beta:f(end) and delete all elements of v that are elements of f.
            vector<int> v;
            for (int j = beta; j <= f.back(); j++) {
                if (find(f.begin(), f.end(), j) == f.end()){
                    v.push_back(j);
                }

            }

            //f!=g"
            vector<int> fValNotG = vector<int>();
            vector<int> gValNotF = vector<int>();

            //f(f!=g)
            set_difference(f.begin(), f.end(), g_indices.begin(), g_indices.end(), back_inserter(fValNotG));

            // g(f!=g)
            set_difference(g_indices.begin(), g_indices.end(), f.begin(), f.end(), back_inserter(gValNotF));

            //Concatenate gValNotF to ILST and fValNotG to IFST
            ILST->insert(ILST->end(), gValNotF.begin(), gValNotF.end());
            IFST->insert(IFST->end(), fValNotG.begin(), fValNotG.end());


            //q(g(end) + 1: f(end)) = q(v)
            vector<int> oldM = m;
            int vIdx = 0;
            for(int ii = g_indices.back() + 1; ii <= f.back(); ii++){
                m[ii] = oldM[v[vIdx++]];
            }
            //q[g] = [i,...,i]
            skippedIdx += g_indices.back() - g_indices[0];
            for(int ii = g_indices[0]; ii <= g_indices.back(); ii++){
                m[ii] = i;
            }
            beta += h[i];
        }
    }

    vector<int> current_group;
    current_group.push_back(0);


    vector<vector<int>> ind = vector<vector<int>>();
    for (int i = 1; i < m.size(); ++i) {
        if (m[i] != m[i - 1]) {
            ind.push_back(current_group);
            current_group.clear();
        }
        current_group.push_back(i);
    }

    ind.push_back(current_group);

    return ind;
}



vector<int> blocking(int size, lpck_c * diag, double delta) {

    vector<int> select = vector<int>(size);

    int maxM = -1;
    for(int i = 0; i < size; i++) {

        //if the element has not been selected yet.
        if(select[i] == 0) {
            select[i] = ++maxM;
        }

        for(int j = i + 1; j < size; j++) {
            if(select[i] != select[j]){
                if(lpck_abs(lpck_z_sub(diag[i], diag[j])) <= delta){
                    if(select[j] == 0)
                        select[j] = select[i];
                    else {
                        int p = max(select[i], select[j]);
                        int q = min(select[i], select[j]);
                        for(int k = 0; k < size; k++) {
                            if(select[k] == p) 
                                select[k] = q;
                            if(select[k] > p) 
                                select[k] = select[k] - 1;
                        }
                        maxM--;
                    }
                }
            }
        }
    }
    return select;
}

vector<vector<int>> schurDecomposition(lpck_c ** lapacke_A, lpck_c ** U, int size) {

    lpck_c * w = (lpck_c *) calloc(size, sizeof(lpck_c));

    vector<int> blockList;

    //complex Schur Decomposition
    LAPACKE_zhseqr(LAPACK_ROW_MAJOR, 'S', 'I', size, 1, size, 
                    *lapacke_A, size, w, *U, size);

    vector<int> clusters = blocking(size, w, 0.1);

    vector<int> ILST = vector<int>();
    vector<int> IFST = vector<int>();

    vector<vector<int>> ind = swapping(clusters, &ILST, &IFST);

    for(int i = 0; i < ILST.size(); i++) {
        LAPACKE_ztrexc(LAPACK_ROW_MAJOR, 'V', size, *lapacke_A, size, *U, size, IFST[i] + 1, ILST[i] + 1);
    }

    free(w);

    return ind;
}