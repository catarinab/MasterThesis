#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "utils/distr_mtx_ops.cpp"
#include "utils/helpProccess.cpp"

using namespace std;
//cg -> krylov subspace method // exp
#define epsilon 0.0000000001 

bool debugMtr = false;


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

    int maxIter = size*size;

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
        MPI_Recv(&helpSize, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        switch (status.MPI_TAG) {
            case ENDTAG:
                if((me + 1) != nprocs) 
                    MPI_Send(&me, 1, MPI_DOUBLE, me + 1, ENDTAG, MPI_COMM_WORLD);
                return x;
            case IDLETAG:
                if(status.MPI_SOURCE != 0) break;
                helpProccess(status.MPI_SOURCE, A, b, me, size, helpSize, nprocs, displs, counts);
                break;
            default:
                break;
        }
    }
    
    for(int t = 0; t < maxIter; t++) {
        //cout << "Iteration number: " << t << endl;

        
        //nada que possamos guardar para as proxs contas
        //mas podemos usar o g da iteracao anterior
        denom1 = distrDotProduct(g, g, size, me, nprocs);            

        //nada para reutilizar
        op = distrMatrixVec(A, x, size, me, nprocs);
        if(debugMtr){
            op.printAttr("Ax(t-1)");
        }
            

        

        //g(t) = Ax(t-1) - b
        g =  distrSubOp(op, b, size, me, nprocs);
        if(debugMtr)
            g.printAttr("g");
        


        //g(t)^T g(t)
        //guardar o g para a proxima conta !!
        num1 = distrDotProduct(g, g, size, me, nprocs);
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
                d = subtractVec(op, g, 0, size);
        }

        if(debugMtr) {
            op.printAttr("d*(num1/denom1)");
            d.printAttr("dir");
        }

        //d(t)^T g(t)
        //enviar apenas a direcao
        num2 = distrDotProduct(d, g, size, me, nprocs);
        if(debugMtr)
            cout << "Num2: " << num2 << endl;

        //nao precisamos de enviar nada
        op = distrMatrixVec(A, d, size, me, nprocs);
        if(debugMtr)
            op.printAttr("A*d(t)");
        

        //d(t)^T A*d(t)
        //enviar op
        denom2 = distrDotProduct(d, op, size, me, nprocs);

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
    if(nprocs > 1)
        MPI_Send(&finalIter, 1, MPI_DOUBLE, 1, ENDTAG, MPI_COMM_WORLD);
    return x;
}

void processInput(int argc, char* argv[], string * inputFile) {
    for(int i = 0; i < argc; i++) {

        if(string(argv[i]) == "-dm") 
            debugMtr = true;
        
        if(string(argv[i]) == "-f") 
            *inputFile = string(argv[i+1]);
    }
}


int main (int argc, char* argv[]) {
    int me, nprocs;
    int finalIter = -1;
    double exec_time;
    string input_file;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    processInput(argc, argv, &input_file);
    
    //para todos terem a matrix e o b
    CSR_Matrix csr = buildMtx(input_file);
    int size = csr.getSize();
    Vector b(size, readFile_vec("/home/cat/uni/thesis/project/conjugateGradient/Vec/Vec.txt", size));

    MPI_Barrier(MPI_COMM_WORLD);
    exec_time = -omp_get_wtime();
    Vector x = cg(csr, b, size, b, &finalIter); //initial guess: b
    exec_time += omp_get_wtime();

    if(me == 0 && finalIter != size - 1) {
        for(int i = 0; i < size; i++){
            cout << "x[" << i << "]: " << x.values[i] << endl;
        }
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Final iteration: " << finalIter << endl;
    }
    MPI_Finalize();
    return 0;
}