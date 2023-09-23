#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "distributedUtils.cpp"

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
vector<double> cg(vector<vector<double>> A, vector<double> b, int size, vector<double> x, int * finalIter) {
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int dest = (me == nprocs - 1 ? 0 : me + 1);
    int source = (me == 0 ? nprocs - 1 : me - 1);

    vector<double> g(size); //gradient, inicializar na primeira iteraçao?
    vector<double> d(size); //direction
    double s; //step size
    
    //auxiliar
    vector<double> op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    int flag=0;
    bool idle = 0;
    int helpDest = -1;
    int func = MV;
    int helpSize = 0;
    double dotProd;

    vector<double> auxBuf(size);
    vector<double> auxBuf2(size);

    MPI_Status status;
    MPI_Request sendGradReq;
    MPI_Request sendDirReq;
    MPI_Request sendIdleReq;
    if(me == 0) {
        op = distrMatrixVec(x, A, size, me, nprocs);
        d = distrSubOp(b, op, size, me, nprocs); //initial direction = residue
    }

    while(me != 0) {
        MPI_Recv(&auxBuf[0], size, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        switch (status.MPI_TAG) {
            case ENDTAG:
                if(status.MPI_TAG == ENDTAG && g[0] != dest) {
                    MPI_Send(&g[0], 1, MPI_DOUBLE, dest, ENDTAG, MPI_COMM_WORLD);
                    return x;
                }
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
        cout << "============Iteration number:============" << t << endl;

        //g(t-1)^T g(t-1)
        if(t != 0){
            denom1 = distrDotProduct(g, g, size, me, nprocs, dest);
            //Ax(t-1)
            op = distrMatrixVec(x, A, size, me, nprocs);
            
        }
        if(debugMtr){
                cout << "Ax(t-1): " << endl;
                for(int i = 0; i < size; i++) {
                    cout << op[i] << endl;
                }
            }
            

        

        //g(t) = Ax(t-1) - b
        g =  distrSubOp(op, b, size, me, nprocs);
        if(debugMtr) {
            cout << "g(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << g[i] << endl;
            }
        }
        
        //g(t)^T g(t)
        num1 = distrDotProduct(g, g, size, me, nprocs, dest);
        if(debugMtr) cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon){
            *finalIter = t;
            if(nprocs > 1){
                MPI_Send(&num1, 1, MPI_DOUBLE, 1, ENDTAG, MPI_COMM_WORLD);
            }
            return x;
        } 

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        //se estivermos na iteracao 0, a direcao foi ja calculada no inicio da funcao com o residuo
        //senao, temos de receber a direcao da iteracao anterior e calcular a nova direcao
        if(t!= 0){
            #pragma omp parallel for
            for(int i = 0; i < size; i++) {
                d[i] = -g[i] + (num1/denom1) * d[i];
                if(debugMtr) {
                cout << "-g[i]: " << -g[i] << endl;
                cout << "(num1/denom1): " << (num1/denom1) << endl;
                cout << "d[i]: " << d[i] << endl;
                cout << "d(t)[" << i << "]: " << d[i] << endl;
            }
                
            }
        }

        //d(t)^T g(t)
        num2 = distrDotProduct(d, g, size, me, nprocs, dest);

        //A*d(t)
        op = distrMatrixVec(d, A, size, me, nprocs);

        //d(t)^T A*d(t)
        denom2 = distrDotProduct(d, op, size, me, nprocs, dest);

        s = -num2/denom2;

        if(debugMtr){
            cout << "num2: " << num2 << endl;
            cout << "denom2: " << denom2 << endl;
            cout << "s: " << s << endl;
        }

        #pragma omp parallel for
        for(int i = 0; i < size; i++) {
            x[i] = x[i] + s*d[i];
        }

        if(debugMtr) {
            cout << "x(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << x[i] << endl;
            }
        }
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

    processInput(argc, argv);

    //dividir trabalho pelos nodes
    vector<vector<double>> A = buildMatrix(input_file);
    vector<double> b = buildVector(input_file);
    int size = b.size();
    //num max de threads
    omp_set_num_threads(size/2);
    //initial guess: b
    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    vector<double> x = cg(A, b, size, b, &finalIter);
    exec_time += omp_get_wtime();
    if(me == 0) {
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Final iteration: " << finalIter << endl;
        for(int i = 0; i < size; i++) {
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    }
    cout << "Proccess number: " << me << " Ending execution" << endl;
    MPI_Finalize();
    return 0;
}