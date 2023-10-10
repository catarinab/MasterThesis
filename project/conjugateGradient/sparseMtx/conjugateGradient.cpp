#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "../../utils/distr_mtx_ops.cpp"
#include "../../utils/helpProccess.cpp"

using namespace std;
#define epsilon 0.00001 

bool debugMtr = false;
int maxIter;

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

    maxIter = size*size;

    initGatherVars(size, nprocs);

    int dest = 0;

    Vector g(size); //gradient
    Vector d(size); //direction
    double s; //step size
    
    //auxiliar
    Vector op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    int helpSize = 0;
    double dotProd;

    MPI_Status status;
    
    if(me == 0) {
        op = distrMatrixVec(A, x, size, me, nprocs);
        d = subtractVec(b, op, 0, size); //initial direction = residue
        if(debugMtr){
            op.printAttr("A*x0");
            d.printAttr("dir0");
        }
        
    }      
    while(me != 0) {
        MPI_Bcast(&helpSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(helpSize == ENDTAG) return x;
        else if(helpSize > 0)
            helpProccess(A, b, me, size, helpSize, nprocs, displs, counts);
    }
    
    for(int t = 0; t < maxIter; t++) {
        //cout << "Iteration number: " << t << endl;

        //g(t-1)^T g(t-1)
        denom1 = distrDotProduct(g, g, size, me, nprocs);            
        
        op = distrMatrixVec(A, x, size, me, nprocs);
        if(debugMtr){
            op.printAttr("Ax(t-1)");
        }

        //g(t) = Ax(t-1) - b
        g =  distrSubOp(op, b, size, me, nprocs);
        if(debugMtr)
            g.printAttr("g");
        

        //g(t)^T g(t)
        num1 = distrDotProduct(g, g, size, me, nprocs);
        if(debugMtr)
            cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon) {
            *finalIter = t;
            int temp = ENDTAG;
            MPI_Bcast(&temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return x;
        } 

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        if(t!= 0){
                op = d*(num1/denom1);
                d = subtractVec(op, g, 0, size);
        }

        if(debugMtr) {
            op.printAttr("d*(num1/denom1)");
            d.printAttr("dir");
        }

        //d(t)^T g(t)
        num2 = distrDotProduct(d, g, size, me, nprocs);
        if(debugMtr)
            cout << "Num2: " << num2 << endl;


        op = distrMatrixVec(A, d, size, me, nprocs);
        if(debugMtr)
            op.printAttr("A*d(t)");
        

        //d(t)^T * A*d(t)
        denom2 = distrDotProduct(d, op, size, me, nprocs);

        //s = -(d(t)^T g(t))/(d(t)^T * A*d(t))
        s = -num2/denom2;
        if(debugMtr) {
            cout << "num2: " << num2 << endl;
            cout << "denom2: " << denom2 << endl;
            cout << "s: " << s << endl;
        }
        

        op = d*s;

        if(debugMtr)
            op.printAttr("d*s");
        
        //x(t) = x(t-1) + s(t)d(t)
        x = distrSumOp(x, op, size, me, nprocs);

        if(debugMtr) 
            x.printAttr("x iter: "+to_string(t));
        

    }
    
    *finalIter = (maxIter - 1);
    cout << "Error. Max iterations reached, system did not converge" << endl;
    int temp = ENDTAG;
    MPI_Bcast(&temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return x;
}

void processInput(int argc, char* argv[], string * inputFile, string * inputFileVec) {
    for(int i = 0; i < argc; i++) {

        if(string(argv[i]) == "-dm") 
            debugMtr = true;
        
        if(string(argv[i]) == "-fm") 
            *inputFile = string(argv[i+1]);
        if(string(argv[i]) == "-fv") 
            *inputFileVec = string(argv[i+1]);
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    int finalIter = -1;
    double exec_time;
    string input_file;
    string input_fileVec;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    processInput(argc, argv, &input_file, &input_fileVec);
    
    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx(input_file);
    int size = csr.getSize();
    Vector b(size, readFile_vec(input_fileVec, size));

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    Vector x = cg(csr, b, size, b, &finalIter); //initial guess: b
    exec_time += omp_get_wtime();

    if(me == 0) {
        for(int i = 0; i < size && finalIter != (maxIter - 1); i++){
            cout << "x[" << i << "]: " << x.values[i] << endl;
        }
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Final iteration: " << finalIter << endl;
    }

    free(displs);
    free(counts);
    MPI_Finalize();
    return 0;
}