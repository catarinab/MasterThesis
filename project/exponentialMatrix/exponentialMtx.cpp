#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "../utils/distr_mtx_ops.cpp"
#include "../utils/helpProccess.cpp"
#include "../utils/scalar_ops.cpp"

using namespace std;
#define epsilon 1e-12 //10^-12
#define omega 0.0001

bool debugMtr = false;
bool vecFile = false;
int beta = 1;

/*Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : An m × m array. (CSR_Matrix)
    b : Initial mx1 (Vector).
    n : One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1 (int)
    m : Dimension of the matrix (int)

    Returns
    -------
    Q : An m x (n + 1) array (dense_matrix rows:m cols: n+1), where the columns are an orthonormal basis of the Krylov subspace.
    H : An (n + 1) x n array (dense_matrix rows: n+1, cols: n). A on basis Q. It is upper Hessenberg.
    */
int arnoldiIteration(CSR_Matrix A, Vector b, int n, int m, int me, int nprocs, dense_Matrix * V, dense_Matrix * H) {
    int helpSize = 0;
    int sendEnd = ENDTAG;

    //helper nodes
    while(me != 0) {
        MPI_Bcast(&helpSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(helpSize == ENDTAG) return 0;
        else if(helpSize > 0)
            helpProccess(A, me, m, helpSize, nprocs, displs, counts);
    }

    b = b / b.getNorm2();
    V->setCol(0, b);

    int k = 1;

    //auxiliar
    Vector opResult(m);

    for(k = 1; k < n + 1 ; k++) {

        Vector w = distrMatrixVec(A, V->getCol(k-1), m, me, nprocs);


        for(int j = 0; j < k; j++) {
            H->setValue(j, k-1, distrDotProduct(w, V->getCol(j), m, me, nprocs));
            opResult = V->getCol(j) * H->getValue(j, k-1);
            w = distrSubOp(w, opResult, m, me, nprocs);
        }

        if(k >= m) break;

        H->setValue(k, k - 1, w.getNorm2());
        if(H->getValue(k, k - 1) != 0)
            V->setCol(k, w / H->getValue(k, k - 1));

        else{
            printf("Krylov subspace exhausted at iteration %d.", k);
            MPI_Bcast(&sendEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return k;
        }
    }
    MPI_Bcast(&sendEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return --k;
}


int findM(dense_Matrix H, int theta , int * power) {
    int m = 1; //2^0
    *power = 0; //2^0
    double norm = H.getNorm2();

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

    double norm1 = A.getNorm2();

    cout << "norm1: " << norm1 << endl;

    vector<dense_Matrix>::iterator ptr = powers->begin();

    *ptr++ = identity;
    *ptr++ = A;
    dense_Matrix A2 = denseMatrixMatrixMult(A, A);
    *ptr++ = A2;
    dense_Matrix A4 = denseMatrixMatrixMult(A2, A2);
    *++ptr = A4;
    dense_Matrix A6 = denseMatrixMatrixMult(A2, A4);
    *(ptr + 2) = A6;


    if(norm1 < theta[0])
        return 3;
    else if(norm1 < theta[1])
        return 5;


    dense_Matrix A8 = denseMatrixMatrixMult(A4, A4);

    powers->push_back(A8);

    if(norm1 < theta[2])
        return 7;
    if(norm1 < theta[3])
        return 9;
    if(norm1 < theta[4])
        return 13;

    //so neste caso fazemos o scaling!
    *s = findM(A, theta[4], power);
    return 13;

}

dense_Matrix padeApprox(dense_Matrix H) {
    vector<dense_Matrix> powers(8);
    int s = 0, power = 0;
    int m = definePadeParams(&powers, &s, &power, H);

    cout << "m: " << m << endl;

    cout << "s: " << s << endl;

    cout << "power: " << power << endl;



    dense_Matrix identity = dense_Matrix(H.getColVal(), H.getRowVal());
    identity.setIdentity();

    dense_Matrix U;
    dense_Matrix V;

    vector<double> coeff = get_pade_coefficients(m);

    //normal pade approx
    if(m!= 13) {
        U = identity * coeff[1];
        V = identity * coeff[0];

        for(int j = m; j >= 3; j-=2) {
            U = denseMatrixMatrixAdd(U, powers[j-1] * coeff[j]);
            V = denseMatrixMatrixAdd(V, powers[j-1] * coeff[j-1]);
        }
    }
    //pade approx with scaling and squaring
    if(m == 13){
        H = H/s;
        H.printAttr("H");
        dense_Matrix op1 = denseMatrixMatrixAdd(powers[6]*coeff[7], powers[4]*coeff[5]);
        dense_Matrix op2 = denseMatrixMatrixAdd(powers[2]*coeff[3], identity*coeff[1]);
        dense_Matrix sum1 = denseMatrixMatrixAdd(op1, op2);
        op1 = denseMatrixMatrixAdd(powers[6]*coeff[13], powers[4]*coeff[11]);
        op2 = denseMatrixMatrixAdd(op1, powers[2]*coeff[9]);
        dense_Matrix sum2 = denseMatrixMatrixMult(powers[6], op2);
        U = denseMatrixMatrixAdd(sum1, sum2);

        op1 = denseMatrixMatrixAdd(powers[6]*coeff[6], powers[4]*coeff[4]);
        op2 = denseMatrixMatrixAdd(powers[2]*coeff[2], identity*coeff[0]);
        sum1 = denseMatrixMatrixAdd(op1, op2);
        op1 = denseMatrixMatrixAdd(powers[6]*coeff[12], powers[4]*coeff[10]);
        op2 = denseMatrixMatrixAdd(op1, powers[2]*coeff[8]);
        sum2 = denseMatrixMatrixMult(powers[6], op2);
        V = denseMatrixMatrixAdd(sum1, sum2);

    }

    U = denseMatrixMatrixMult(H, U);

    dense_Matrix num1 = denseMatrixMatrixAdd(V, U);
    dense_Matrix num2 = denseMatrixMatrixSub(V, U);
    dense_Matrix num2Inv = denseMatrixInverse(num2);
    num1.printAttr("nom");
    num2Inv.printAttr("denom");
    dense_Matrix res = denseMatrixMatrixMult(num2Inv, num1);
    res.printAttr("res");
    if(s != 0)
        for(int i = 0; i < power; i++)
            res = denseMatrixMatrixMult(res, res);
    return res;
}

int main (int argc, char* argv[]) {
    int me, nprocs;
    double exec_time;
    bool vecFile = false;

    int krylovDegree = 3;
    int finalKrylovDegree;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx("/home/cat/uni/thesis/project/mtx/Trefethen_20b/Trefethen_20b.mtx");
    int size = csr.getSize();
    Vector b(size);
    b.getOnesVec();

    initGatherVars(size, nprocs);


    dense_Matrix V(size, krylovDegree);
    dense_Matrix H(krylovDegree, krylovDegree);

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    //from this, we get the Orthonormal basis of the Krylov subspace (V) and the upper Hessenberg matrix (H)
    //finalKrylovDegree = arnoldiIteration(csr, b, krylovDegree, size, me, nprocs, &V, &H);

    MPI_Barrier(MPI_COMM_WORLD);


    if(me == 0) {

        H.setRandomSmall();
        H.printAttr("H");

        dense_Matrix res = padeApprox(H);
        res.printAttr("expH");


        exec_time += omp_get_wtime();
        cout << "exec_time: " << exec_time << endl;

    }



    free(displs);
    free(counts);
    MPI_Finalize();
    return 0;
}