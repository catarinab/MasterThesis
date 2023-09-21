#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

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

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix(string input_file) {
    vector<vector<double>> A{
        {10, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {1, 20, 1, 2, 3, 4, 5, 6, 7, 8},
        {2, 1, 30, 1, 2, 3, 4, 5, 6, 7},
        {3, 2, 1, 40, 1, 2, 3, 4, 5, 6},
        {4, 3, 2, 1, 50, 1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1, 60, 1, 2, 3, 4},
        {6, 5, 4, 3, 2, 1, 70, 1, 2, 3},
        {7, 6, 5, 4, 3, 2, 1, 80, 1, 2},
        {8, 7, 6, 5, 4, 3, 2, 1, 90, 1},
        {9, 8, 7, 6, 5, 4, 3, 2, 1, 100}
    };
    return A;
}

//construir tendo em conta as matrizes do matlab?
vector<double> buildVector(string input_file) {
    vector<double> b{2, 8, 9, 2, 3, 4, 5, 6, 7, 8};
    return b;
}

vector<double> subtractVec(vector<double> a, vector<double> b, int begin, int end) {
    vector<double> res(end - begin);
    int resIndex = 0;

    #pragma omp parallel for private(resIndex)
    for (int i = begin; i < end; i++) {
        resIndex = i - begin;
        res[resIndex] = a[i] - b[i];
    }
    return res;
}

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int begin, int end, int size) {
    vector <double> res(end - begin);
    int resIndex = 0;
    
    #pragma omp parallel for private(resIndex)
	for (int i = begin; i < end; i++) {
        resIndex = i - begin;
		res[resIndex] = 0;
		for (int j = 0; j < size; j++) {
			res[resIndex] += matrix[i][j] * v[j];
		}
	}
	return res;
}

double dotProduct(vector<double> a, vector<double> b, int begin, int end) {
	double dotProd = 0.0;

    #pragma omp parallel for reduction(+:dotProd)
	for (int i = begin; i < end; i++) {
		dotProd += (a[i] * b[i]);
	}
    return dotProd;
}

void sendVectors(vector<double> a, vector<double> b, int begin, int helpSize, int dest, int func, int me, int size) {

    MPI_Send(&func, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);
    MPI_Send(&helpSize, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);

    if(func == VV || func == SUB) {
        MPI_Send(&a[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&b[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
    }
    else if(func == MV)
        MPI_Send(&a[begin], size, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);

}


double distrDotProduct(vector<double> a, vector<double> b, int size, int me, int nprocs, int dest) {
    int count = 0; 
    int flag = 0;
    int helpSize = size/nprocs;
    int end = 0;

    while(count != nprocs - 1) {
        MPI_Send(&helpSize, 1, MPI_DOUBLE, dest, IDLETAG, MPI_COMM_WORLD);
        sendVectors(a, b, count * helpSize, helpSize, dest, VV, me, size);
        count++;
        dest = (dest == nprocs - 1 ? 0 : dest + 1);
    }

    if(count == 0) end = size;
    else end = size - helpSize + 1;
    
    double dotProd = dotProduct(a, b, count * helpSize, size);

    double temp = 0;
    dest = 1;

    while(count > 0) {
        MPI_Recv(&temp, 1, MPI_DOUBLE, dest, VV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dotProd += temp;
        count--; dest++;
    }

    return dotProd;
}

vector<double> distrSubOp(vector<double> a, vector<double> b, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    int end = 0;
    int dest = 1;
    double temp[helpSize];
    
    vector<double> res;
    vector<double> finalRes;   

    while(count != nprocs - 1) {
        MPI_Send(&helpSize, 1, MPI_DOUBLE, dest, IDLETAG, MPI_COMM_WORLD);
        sendVectors(a, b, count * helpSize, helpSize, dest, SUB, me, size);
        count++; dest++;
    }

    if(helpSize*(nprocs-1) != size || count == 0)
        res = subtractVec(a, b, count * helpSize, size);

    dest = 1;
    while(count > 0) {
        MPI_Recv(&temp, helpSize, MPI_DOUBLE, dest, SUB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < helpSize; i++) {
            finalRes.push_back(temp[i]);
        }
        dest++; count--;
    }
    
    if(helpSize*(nprocs-1) != size || count == 0) {
        for(int i = 0; i < res.size(); i++) {
            finalRes.push_back(res[i]);
        }
    }
    return finalRes;
}


vector<double> distrMatrixVec(vector<double> vec, vector<vector<double>> A, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    if(helpSize == 0) helpSize = 1; // caso extremo em que o numero de procs e maior que o tamanho do vetor

    int end = 0;
    int dest = 1;
    double temp[helpSize];
    
    vector<double> res;
    vector<double> finalRes;   

    while(count != nprocs - 1 && count != size) {
        MPI_Send(&helpSize, 1, MPI_DOUBLE, dest, IDLETAG, MPI_COMM_WORLD);
        sendVectors(vec, vector<double>(0), 0, helpSize, dest, MV, me, size);

        int begin = count * helpSize;
        int sendEnd = count * helpSize + helpSize;
        if(sendEnd > size) sendEnd = size;
        MPI_Send(&begin, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&sendEnd, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);

        count++; dest++;
    }

    if(count <= size || count == 0){
        res = matrixVector(A, vec, count * helpSize, size, size);
    }
        
    
        

    dest = 1;
    while(count > 0) {
        MPI_Recv(&temp, helpSize, MPI_DOUBLE, dest, MV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i = 0; i < helpSize; i++) 
            finalRes.push_back(temp[i]);
        dest++; count--;
    }

    if(count != size || count == 0) {
        for(int i = 0; i < res.size(); i++) 
            finalRes.push_back(res[i]);
    }

    return finalRes;
}


void helpProccess(int helpDest, vector<vector<double>> A, vector<double> b, int me, int size) {
    int func = -1;
    int helpSize = 0;
    int flag = 0;
    double dotProd = 0;
    int begin = 0;
    int end = 0;

    MPI_Status status;
    MPI_Request sendIdleReq;

    vector<double> op(0);
    vector<double> auxBuf2(0);


    MPI_Recv(&func, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&helpSize, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);

    vector<double> auxBuf(size);

    MPI_Recv(&auxBuf[0], size, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);


    if(func == VV || func == SUB) {
        auxBuf2.resize(helpSize);
        MPI_Recv(&auxBuf2[0], helpSize, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    }
    switch(func){
        case MV:
            MPI_Recv(&begin, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&end, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            op.resize(helpSize);
            op = matrixVector(A, auxBuf, begin, end, size);
            MPI_Send(&op[0], helpSize, MPI_DOUBLE, helpDest, MV, MPI_COMM_WORLD);
            break;
        case VV:
            dotProd = dotProduct(auxBuf, auxBuf2, 0, helpSize);

            MPI_Send(&dotProd, 1, MPI_DOUBLE, helpDest, VV, MPI_COMM_WORLD);
            break;
        case SUB:
            op.resize(helpSize);
            op = subtractVec(auxBuf, auxBuf2, 0, helpSize);
            MPI_Send(&op[0], helpSize, MPI_DOUBLE, helpDest, SUB, MPI_COMM_WORLD);
            break;
        default: 
            cout << "Proccess number: " << me << " Received wrong function tag from node " << helpDest << endl;
            break;
    }
}

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
        if(debugLD || debugMtr || debugParallel)cout << "============Iteration number:============" << t << endl;

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