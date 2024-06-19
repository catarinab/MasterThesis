#include <iostream>
#include <string>
#include <cstring>
#include <omp.h>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <iomanip>
#include <fstream>

#include "../utils/headers/dense_vector.hpp"
#include "../utils/headers/mtx_ops_mkl.hpp"
#include "../cubature/cubature.h"
#include "../utils/headers/arnoldi_iteration_shared_nu.hpp"
#include "../utils/headers/arnoldi_iteration_shared.hpp"
#include "../utils/headers/fractional_Krylov.hpp"
#include "../utils/headers/scaling_and_squaring.hpp"

using namespace std;
using namespace autodiff;

int sparseMatrixSize;
dense_matrix V;
dense_matrix H;
double normu0;
double t;
autodiff::ArrayXreal q(2);

vector<double> juliares;

void readJuliaVec(const string& filename = "juliares.txt") {
    ifstream inputFile(filename);
    if (inputFile) {
        double value;
        while (inputFile >> value) {
            juliares.push_back(value);
        }
        inputFile.close();
    }
    else {
        cout << "Error opening julia vector file" << endl;
    }
}

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
    #pragma omp parallel for schedule(guided)
    for (unsigned j = 0; j < npts; ++j) { //evaluate the integrand for npts points
        autodiff::ArrayXreal u(3);
        u << x[j*ndim+0], x[j*ndim+1], x[j*ndim+2];

        autodiff::real newU = totalRNG(q, u, t);
        if(newU.val() < 1e5) {
            dense_matrix expHT = scalingAndSquaring(H * -newU.val());
            denseMatrixMult(V, expHT).getCol(0, &fval[j * fdim]);
        }
        else
            memset(&fval[j*fdim], 0, fdim * sizeof(double)); //tends to 0
    }
    return 0; // success
}

void solve(const csr_matrix &A, dense_vector u0, int krylovDegree, double atol = 1e-8, double rtol = 1e-5) {

    t = 1;

    double exec_time_arnoldi, exec_time_uT, exec_time_duTdp;

    normu0 = u0.getNorm2();

    u0 /= normu0;

    double alpha = 0.15135433606127727;
    double gamma = 1;

    int nu = ceil(gamma/alpha);

    V = dense_matrix(sparseMatrixSize, krylovDegree);
    H = dense_matrix(krylovDegree, krylovDegree);

    cerr << "nu: " << nu << endl;

    exec_time_arnoldi = -omp_get_wtime();
    if(nu == 1)
        arnoldiIteration(A, u0, krylovDegree, sparseMatrixSize, &V, &H);
    else
        arnoldiIteration(A, u0, krylovDegree, sparseMatrixSize, &V, &H, nu);
    exec_time_arnoldi += omp_get_wtime();

    cerr << "arnoldi done " << endl;

    q << alpha, gamma;

    vector<double> uT = vector<double>(sparseMatrixSize, 0);
    vector<double> erruT = vector<double>(sparseMatrixSize, 0);

    double xmin[3] = {0,0,0}, xmax[3] = {1,1,1};
    exec_time_uT = -omp_get_wtime();
    hcubature_v(sparseMatrixSize, uTCalcV, nullptr, 3, xmin, xmax, 0, atol, rtol, ERROR_L2, uT.data(), erruT.data());
    exec_time_uT += omp_get_wtime();
    vector<double> diff;
    for(int idx = 0; idx < sparseMatrixSize; idx++) {
        uT[idx] = uT[idx] * normu0;
        diff.push_back(uT[idx] - juliares[idx]);
    }

    double relNorm = cblas_dnrm2(sparseMatrixSize, diff.data(), 1) / cblas_dnrm2(sparseMatrixSize, juliares.data(), 1);

    cout << exec_time_uT << "," << 0 << "," << relNorm << endl;
}


void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxPath, double * rtol, string * juliaPath) {

    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxPath = "A-" + std::string(argv[i + 1]) + ".mtx";
            *juliaPath = "juliares-" + std::string(argv[i + 1]) + ".txt";
        }
        else if(strcmp(argv[i], "-err") == 0) {
            *rtol = stod(argv[i+1]);
        }
    }
}


int main (int argc, char* argv[]) {
    //input values

    string mtxPath;
    string juliaPath;
    int krylovDegree;
    double rtol;

    mkl_domain_set_num_threads(1, MKL_DOMAIN_BLAS);
    mkl_domain_set_num_threads(omp_get_max_threads(), MKL_DOMAIN_LAPACK);

    cerr << mkl_domain_get_max_threads(MKL_DOMAIN_BLAS) << endl;
    cerr << mkl_domain_get_max_threads(MKL_DOMAIN_LAPACK) << endl;

    if(argc != 7){
        cerr << "Usage: " << argv[0] << " -s <size> -k <krylov-degree> -m <mtxSize> -err <rtol>" << endl;
        return 1;
    }

    processArgs(argc, argv, &krylovDegree, &mtxPath, &rtol, &juliaPath);
    readJuliaVec(juliaPath);

    //initializations of needed matrix and vectors
    csr_matrix A = buildFullMatrix(mtxPath);
    sparseMatrixSize = (int) A.getSize();

    dense_vector b = dense_vector(sparseMatrixSize);
    b.insertValue(0, 1);

    solve(A, b, krylovDegree, 1e-8, rtol);

    mkl_sparse_destroy(A.getMKLSparseMatrix());

    return 0;
}