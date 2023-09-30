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
#define IDLETAG 1

bool debugMtr = false;
string input_file;



/* Conjugate Gradient Method: iterative method for efficiently solving linear systems of equations: Ax=b
    Step 1: Compute gradient: g(t) = Ax(t - 1) - b
    Step 2: if g(t)^T g(t) < epsilon => return
    Step 3: Compute direction vector: d(t)
    Step 4: Compute step size: s(t)
    Step 5: Compute new aproximation: x(t) = x(t-1) + s(t)d(t)
*/
Vector cg(CSR_Matrix A, Vector b, int size, Vector x, int * finalIter) {
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int dest = 0;

    Vector g(size); //gradient, inicializar na primeira iteraçao?
    Vector d(size); //direction
    double s; //step size
    
    //auxiliar
    Vector op(size);
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
        op = distrMatrixVec(A, x, size, me, nprocs);    
        //enviar op    
        d = distrSubOp(b, op, size, me, nprocs); //initial direction = residue
        if(debugMtr){
            op.printAttr("A*x0");
            d.printAttr("dir0");
        }
        
    }
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
    for(int t = 0; t < MAXITER; t++) {
        cout << "Iteration number: " << t << endl;

        //g(t-1)^T g(t-1)
        if(t != 0){
            denom1 = dotProduct(g, g, 0, size);
            //Ax(t-1)            
        }

        op = distrMatrixVec(A, x, size, me, nprocs);
        if(debugMtr){
            op.printAttr("Ax(t-1)");
        }
            

        

        //g(t) = Ax(t-1) - b
        //enviar o op
        g =  distrSubOp(op, b, size, me, nprocs);
        if(debugMtr) {
            g.printAttr("g");
        }


        //g(t)^T g(t)
        num1 = dotProduct(g, g, 0, size);
        if(debugMtr)
            cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon) {
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
                //enviar ambos...
                d = distrSubOp(op, g, size, me, nprocs);
                if(debugMtr) {
                    op.printAttr("d*(num1/denom1)");
                    d.printAttr("dir");
                }
        }

        //d(t)^T g(t)
        //enviar apenas a direcao
        num2 = dotProduct(d, g, 0, size);
        if(debugMtr)
            cout << "Num2: " << num2 << endl;

        //A*d(t)
        op = distrMatrixVec(A, d, size, me, nprocs);
        if(debugMtr){
            op.printAttr("A*d(t)");
        }

        //d(t)^T A*d(t)
        //enviar ambos
        denom2 = dotProduct(d, op, 0, size);

        s = -num2/denom2;
        if(debugMtr){
            cout << "num2: " << num2 << endl;
            cout << "denom2: " << denom2 << endl;
            cout << "s: " << s << endl;
        }
        

        op = d*s;

        if(debugMtr)
            op.printAttr("d*s");
        
        //x(t) = x(t-1) + s(t)d(t)
        x = addVec(x, op, 0, size);

        if(debugMtr) 
            x.printAttr("x iter: "+to_string(t));
        

    }
    *finalIter = (MAXITER - 1);
    if(nprocs > 1)
        MPI_Send(&finalIter, 1, MPI_DOUBLE, 1, ENDTAG, MPI_COMM_WORLD);
    return x;
}

void processInput(int argc, char* argv[]) {
    for(int i = 0; i < argc; i++) {

        if(string(argv[i]) == "-dm") 
            debugMtr = true;
        
        if(string(argv[i]) == "-f") 
            input_file = string(argv[i+1]);
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    double master = -1;
    int finalIter = -1;
    double exec_time;       /* execution time */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    processInput(argc, argv);

    //dividir trabalho pelos nodes
    
    CSR_Matrix csr = buildMtx(input_file);
    int size = csr.getSize();
    Vector b(size, true);

    //num max de threads
    omp_set_num_threads(size/2);
    //initial guess: b
    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    Vector x = cg(csr, b, size, b, &finalIter);
    exec_time += omp_get_wtime();
    double sum = 0;
    if(me == 0) {
        for(int i = 0; i < size; i++) sum += x.values[i];
        cout << "Sum: " << sum << endl;
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Final iteration: " << finalIter << endl;
    }
    MPI_Finalize();
    return 0;
}