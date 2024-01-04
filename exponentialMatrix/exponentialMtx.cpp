#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <cmath>

#include "../utils/headers/pade_exp_approx_utils.hpp"
#include "../utils/headers/help_proccess.hpp"
#include "../utils/headers/distr_mtx_ops.hpp"
#include "../utils/headers/mtx_ops.hpp"

using namespace std;

double betaVal = 1;

/*Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : An m × m array (csr_matrix).
    b : Initial mx1 vector (dense_vector).
    n : One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1 (int)
    m : Dimension of the matrix (int)

    Returns
    -------
    V : An m x n array (dense_matrix rows:m cols: n+1), where the columns are an orthonormal basis of the Krylov subspace.
    H : An n x n array (dense_matrix rows: n+1, cols: n). A on basis Q. It is upper Hessenberg.
    */
int arnoldiIteration(csr_matrix A, dense_vector b, int k_total, int m, int me, int nprocs, dense_matrix * V,
                        dense_matrix * H) {

    int func = 0;
    int sendEnd = ENDTAG;

    //helper nodes
    while(me != 0) {
        MPI_Bcast(&func, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(func == ENDTAG) return 0;
        else if(func > 0)
            helpProccess(A, me, m, func, nprocs, displs, counts);
    }

    V->setCol(0, b);

    betaVal = b.getNorm2();

    int k = 1;

    //auxiliar
    dense_vector opResult(m);
    dense_vector w(m);

    for(k = 1; k < k_total + 1; k++) {
        w = distrMatrixVec(A, V->getCol(k-1), m, me, nprocs);

        for(int j = 0; j < k; j++) {
            H->setValue(j, k-1, distrDotProduct(w, V->getCol(j), m, me, nprocs));
            opResult = V->getCol(j) * H->getValue(j, k-1);
            w = distrSubOp(w, opResult, m, me, nprocs);
        }

        
        if( k == k_total) break;
        H->setValue(k, k - 1, w.getNorm2());

        if(H->getValue(k, k - 1) != 0) 
            V->setCol(k, w / H->getValue(k, k - 1));
    }
    MPI_Bcast(&sendEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return k;
}

//Calculate the Pade approximation of the exponential of matrix H.
dense_matrix padeApprox(dense_matrix H) {
    vector<dense_matrix> powers(8);
    int s = 0, twoPower = 0, m = 0;
    H = definePadeParams(&powers, &m, &s, H);

    dense_matrix identity = dense_matrix(H.getColVal(), H.getRowVal());
    identity.setIdentity();

    dense_matrix U = dense_matrix(H.getColVal(), H.getRowVal());
    dense_matrix V = dense_matrix(H.getColVal(), H.getRowVal());

    vector<double> coeff = get_pade_coefficients(m);


    if(m!= 13) {
        U = identity * coeff[1];
        V = identity * coeff[0];

        for(int j = m; j >= 3; j-=2) {
            U = denseMatrixAdd(U, powers[j-1] * coeff[j]);
            V = denseMatrixAdd(V, powers[j-1] * coeff[j-1]);
        }
    }

    if(m == 13){
        dense_matrix op1 = denseMatrixAdd(powers[6]*coeff[7], powers[4]*coeff[5]);
        dense_matrix op2 = denseMatrixAdd(powers[2]*coeff[3], identity*coeff[1]);
        dense_matrix sum1 = denseMatrixAdd(op1, op2);
        op1 = denseMatrixAdd(powers[6]*coeff[13], powers[4]*coeff[11]);
        op2 = denseMatrixAdd(op1, powers[2]*coeff[9]);
        dense_matrix sum2 = denseMatrixMult(powers[6], op2);
        U = denseMatrixAdd(sum1, sum2);

        op1 = denseMatrixAdd(powers[6]*coeff[6], powers[4]*coeff[4]);
        op2 = denseMatrixAdd(powers[2]*coeff[2], identity*coeff[0]);
        sum1 = denseMatrixAdd(op1, op2);
        op1 = denseMatrixAdd(powers[6]*coeff[12], powers[4]*coeff[10]);
        op2 = denseMatrixAdd(op1, powers[2]*coeff[8]);
        sum2 = denseMatrixMult(powers[6], op2);
        V = denseMatrixAdd(sum1, sum2);

    }


    U = denseMatrixMult(H, U);

    dense_matrix num1 = denseMatrixSub(V, U);
    dense_matrix num2 = denseMatrixAdd(V, U);

    dense_matrix res = solveEq(num1, num2);

    if(s != 0)
        for(int i = 0; i < s; i++)
            res = denseMatrixMult(res, res);

    return res;
}

//Process input arguments
void processArgs(int argc, char* argv[], int * krylovDegree, string * mtxName, double * normVal) {
    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-k") == 0) {
            *krylovDegree = stoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0) {
            *mtxName = argv[i+1];
        }
        else if(strcmp(argv[i], "-n") == 0) {
            *normVal = stod(argv[i+1]);
        }
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double exec_time_pade;
    double exec_time_arnoldi;
    double exec_time;

    //default values
    int krylovDegree = 20; 
    string mtxPath;
    double normVal = 0;
    processArgs(argc, argv, &krylovDegree, &mtxPath, &normVal);

    int finalKrylovDegree;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    //each node will have matrix A and vector b
    
    csr_matrix A = buildMtx(mtxPath);
    int size = A.getSize();

    dense_vector b(size);
    b.insertValue(floor(size/2), 1);

    initGatherVars(size, nprocs);


    dense_matrix V(size, krylovDegree);
    dense_matrix H(krylovDegree, krylovDegree);

    dense_vector unitVec = dense_vector(krylovDegree);
    unitVec.insertValue(0, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    exec_time_arnoldi = -omp_get_wtime();

    arnoldiIteration(A, b, krylovDegree, size, me, nprocs, &V, &H);

    MPI_Barrier(MPI_COMM_WORLD);


    //root node performs pade approximation and outputs results
    if(me == 0) {

        exec_time_arnoldi += omp_get_wtime();

        cout << "exec_time_arnoldi: " << exec_time_arnoldi << endl;

        exec_time_pade = -omp_get_wtime();

        dense_matrix expH = padeApprox(H);

        exec_time_pade += omp_get_wtime();
        cout << "exec_time_pade: " << exec_time_pade << endl;

        dense_matrix op1 = denseMatrixMult(V*betaVal, expH);
        dense_vector res = denseMatrixVec(op1, unitVec);

        double resNorm = res.getNorm2();

        printf("diff: %.15f\n", abs(normVal - resNorm));

        printf("2Norm: %.15f\n", resNorm);

        exec_time += omp_get_wtime();
        cout << "exec_time: " << exec_time << endl;

    }

    free(displs);
    free(counts);

    MPI_Finalize();

    return 0;
}