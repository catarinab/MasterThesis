#include <iostream>
#include <string>
#include <cstring>
#include <omp.h>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <iomanip>

#include "../utils/headers/dense_vector.hpp"
#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../cubature/cubature.h"
#include "../utils/headers/arnoldiIteration-shared-nu.hpp"
#include "../utils/headers/arnoldiIteration-shared.hpp"
#include "../utils/headers/fractionalKrylov.hpp"
#include "../utils/headers/pade_exp_approx.hpp"

using namespace std;
using namespace autodiff;


int sparseMatrixSize;
dense_matrix V;
dense_matrix H;
double normu0;
double t;

autodiff::ArrayXreal q(2);

//hcubature(u->(V*(-H)*exp_cutoff(TotalRNG([α;γ],u),-H))[:,1]*dTotalRNGdp([α;γ],u)',[0;0;0],[1;1;1],atol=atol,rtol=rtol)[1]
int duTdpCalcV(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval) {
    //integrar em u
    #pragma omp parallel for schedule(dynamic)
    for (unsigned j = 0; j < npts; ++j) { //evaluate the integrand for npts points
        autodiff::ArrayXreal u(3);
        u << x[j*ndim+0], x[j*ndim+1], x[j*ndim+2];

        autodiff::real newU = totalRNG(q, u, t);
        if(newU.val() < 1e5) {
            dense_vector temp(sparseMatrixSize);

            dense_vector dRNG = dTotalRNGp(q(0).val(), q(1).val(), u(0), u(1), u(2), t);

            dense_matrix m1 = denseMatrixMult(V, -H);
            dense_matrix expH = scalingAndSquaring(H * -newU.val());
            dense_matrix m2 = denseMatrixMult(m1, expH);
            m2.getCol(0, &temp);

            //temp -> sparseMatrixSize x 1
            //result -> 1 x 2
            //fval -> sparseMatrixSize x 2
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sparseMatrixSize, 2, 1, 1,
                        temp.values.data(), 1, dRNG.values.data(), 2, 0.0, &fval[j*fdim], 2);
        }
        else
            memset(&fval[j*fdim], 0, fdim * sizeof(double));
    }
    return 0; // success
}

int uTCalcV(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval) {
    //integrar em u
    #pragma omp parallel for schedule(dynamic)
    for (unsigned j = 0; j < npts; ++j) { //evaluate the integrand for npts points
        autodiff::ArrayXreal u(3);
        u << x[j*ndim+0], x[j*ndim+1], x[j*ndim+2];

        autodiff::real newU = totalRNG(q, u, t);
        if(newU.val() < 1e5)
            denseMatrixMult(V, scalingAndSquaring(H * -newU.val())).getCol(0, &fval[j*fdim]);
        else
            memset(&fval[j*fdim], 0, fdim * sizeof(double));
    }
    return 0; // success
}

void solve(const csr_matrix &A, dense_vector u0, int krylovDegree, double atol = 1e-8, double rtol = 1e-3) {

    t = 1;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> normalDistribution(0.0, 1.0);

    double exec_time_arnoldi, exec_time_uT, exec_time_duTdp;

    normu0 = u0.getNorm2();

    double alpha = 0.8;
    double gamma = 1;


    /*double alpha = abs(normalDistribution(generator)) * 0.9 + 0.1;
    double gamma  = abs(normalDistribution(generator)) * 0.9 + 0.1;*/

    int nu = ceil(gamma/alpha);

    V = dense_matrix(sparseMatrixSize, krylovDegree);
    H = dense_matrix(krylovDegree, krylovDegree);

    cout << nu << endl;
    exec_time_arnoldi = -omp_get_wtime();
    if(nu == 1)
        arnoldiIteration(A, u0, krylovDegree, sparseMatrixSize, &V, &H);
    else
        arnoldiIteration(A, u0, krylovDegree, sparseMatrixSize, &V, &H, nu);
    exec_time_arnoldi += omp_get_wtime();

    q << alpha, gamma;

    vector<double> uT = vector<double>(sparseMatrixSize, 0);
    vector<double> erruT = vector<double>(sparseMatrixSize, 0);

    double xmin[3] = {0,0,0}, xmax[3] = {1,1,1};

    exec_time_uT = -omp_get_wtime();
    hcubature_v(sparseMatrixSize, uTCalcV, nullptr, 3, xmin, xmax, 0, atol, rtol, ERROR_L2, uT.data(), erruT.data());
    exec_time_uT += omp_get_wtime();
    for(int i = 0; i < sparseMatrixSize; i++)
        uT[i] = uT[i] * normu0;

    vector<double> duTdp(sparseMatrixSize * 2, 0);
    vector<double> errduTdp(sparseMatrixSize * 2, 0);

    exec_time_duTdp = -omp_get_wtime();
    hcubature_v(sparseMatrixSize * 2, duTdpCalcV, nullptr, 3, xmin, xmax, 0, atol, rtol, ERROR_L2, duTdp.data(), errduTdp.data());
    exec_time_duTdp += omp_get_wtime();
    for(int idx = 0; idx < sparseMatrixSize; idx++) {
        for (int col = 0; col < 2; col++) {
            duTdp[idx * 2 + col] = duTdp[idx * 2 + col] * normu0;
            //cout << duTdp[idx * 2 + col] << " ";
        }
        //cout << endl;
    }

    cout << exec_time_uT << "," << exec_time_duTdp << endl;
}


void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxPath) {

    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxPath = argv[i+1];
        }
    }
}


int main (int argc, char* argv[]) {
    //input values
    double exec_time;

    string mtxPath;
    int krylovDegree;

    if(argc != 5){
        cerr << "Usage: " << argv[0] << " -k <init-krylov-degree> -m <mtxPath> " << endl;
        return 1;
    }

    processArgs(argc, argv, &krylovDegree, &mtxPath);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMtx(mtxPath);
    sparseMatrixSize = (int) A.getSize();

    dense_vector b(sparseMatrixSize);
    b.getOnesVec();

    //solve
    solve(A, b, krylovDegree, 1e-8, 0.934537);


    mkl_sparse_destroy(A.getMKLSparseMatrix());

    return 0;
}