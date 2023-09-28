#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "utils/distributed_ops.cpp"

using namespace std;

#define epsilon 0.001
#define MAXITER 100

#define ENDTAG 0
#define IDLETAG 4
#define FUNCTAG 5
#define MV 6
#define VV 7
#define SUB 8

bool debugMtr = false;
bool debugParallel = false;
bool debugLD = false;
string input_file;



/* Conjugate Gradient Method: iterative method for efficiently solving linear systems of equations: Ax=b
    Step 1: Compute gradient: g(t) = Ax(t - 1) - b
    Step 2: if g(t)^T g(t) < epsilon => return
    Step 3: Compute direction vector: d(t)
    Step 4: Compute step size: s(t)
    Step 5: Compute new aproximation: x(t) = x(t-1) + s(t)d(t)
*/
Sparse_Vec cg(CSR_Matrix A, Sparse_Vec b, int size, Sparse_Vec x, int * finalIter) {
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int dest = 0;

    Sparse_Vec g(size); //gradient, inicializar na primeira iteraçao?
    Sparse_Vec d(size); //direction
    double s; //step size
    
    //auxiliar
    Sparse_Vec op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    int flag=0;
    bool idle = 0;
    int helpSize = 0;
    double dotProd;

    MPI_Status status;
    
    if(me == 0) {
        op = sparseMatrixVector(A, x, 0, size, size);
        d = subtractSparseVec(b, op, 0, size); //initial direction = residue
    }
    /*
    while(me != 0) {
        MPI_Recv(&helpSize, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        switch (status.MPI_TAG) {
            case ENDTAG:
                if((me + 1) != nprocs) 
                    MPI_Send(&me, 1, MPI_DOUBLE, me + 1, ENDTAG, MPI_COMM_WORLD);
                return x;
            case IDLETAG:
                if(status.MPI_SOURCE != 0) break;
                helpProccess(status.MPI_SOURCE, A, b, me, size);
                break;
            default:
                break;
        }
    }
    */
    for(int t = 0; t < MAXITER; t++) {
        cout << "============Iteration number:============" << t << endl;

        //g(t-1)^T g(t-1)
        if(t != 0){
            denom1 = dotProductSparseVec(g, g, 0, size);
            //Ax(t-1)
            op = sparseMatrixVector(A, x, 0, size, size);
            
        }
            

        

        //g(t) = Ax(t-1) - b
        g =  subtractSparseVec(op, b, 0, size);
        
        //g(t)^T g(t)
        num1 = dotProductSparseVec(g, g, 0, size);
        if(debugMtr) cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon){
            *finalIter = t;
            if(nprocs > 1){
                MPI_Send(&me, 1, MPI_DOUBLE, 1, ENDTAG, MPI_COMM_WORLD);
            }
            return x;
        } 

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        //se estivermos na iteracao 0, a direcao foi ja calculada no inicio da funcao com o residuo
        //senao, temos de receber a direcao da iteracao anterior e calcular a nova direcao
        if(t!= 0){
                op = d*(num1/denom1);
                d = subtractSparseVec(op, g, 0, size);
        }

        //d(t)^T g(t)
        num2 = dotProductSparseVec(d, g, 0, size);

        //A*d(t)
        op = sparseMatrixVector(A, d, 0, size, size);

        //d(t)^T A*d(t)
        denom2 = dotProductSparseVec(d, op, 0, size);

        s = -num2/denom2;

        if(debugMtr){
            cout << "num2: " << num2 << endl;
            cout << "denom2: " << denom2 << endl;
            cout << "s: " << s << endl;
        }

        op = d*s;
        //x(t) = x(t-1) + s(t)d(t)
        x = addSparseVec(x, op, 0, size);
    }
    *finalIter = MAXITER;
    if(nprocs > 1)
        MPI_Send(&finalIter, 1, MPI_DOUBLE, 1, ENDTAG, MPI_COMM_WORLD);
    return x;
}

void processInput(int argc, char* argv[]) {
    for(int i = 0; i < argc; i++) {
        if(string(argv[i]) == "-dm") {
            debugMtr = true;
        }
        if(string(argv[i]) == "-f") {
            input_file = string(argv[i+1]);
        }
        if(string(argv[i]) == "-dp") {
            debugParallel = true;
        }
        if(string(argv[i]) == "-dl") {
            debugLD = true;
        }
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double master = -1;
    int finalIter = -1;
    double exec_time;       /* execution time */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    initMPIDatatypes();

    processInput(argc, argv);

    //dividir trabalho pelos nodes
    
    CSR_Matrix csr = buildMtx(input_file);
    int size = csr.getSize();
    Sparse_Vec b = buildRandSparseVec(size);
    
    b.printAttr();

    //num max de threads
    omp_set_num_threads(size/2);
    //initial guess: b
    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    Sparse_Vec x = cg(csr, b, size, b, &finalIter);
    exec_time += omp_get_wtime();
    if(me == 0) {
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Final iteration: " << finalIter << endl;
        for(int i = 0; i < x.nz; i++) {
            cout << "x[" << x.nzValues[i].col << "]: " << x.nzValues[i].value << endl;
        }
    }
    cout << "Proccess number: " << me << " Ending execution" << endl;
    MPI_Finalize();
    return 0;
}