#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define epsilon 0.001
#define MAXITER 100

#define ENDTAG 0
#define DIRTAG 1
#define GRADTAG 2
#define XTAG 3
#define IDLETAG 4

bool debugMtr = false;
bool debugParallel = false;
string input_file;

//dividir a construcao pelos nodes e depois fazer reducao?

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix(string input_file) {
    vector<vector<double>> A{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
    return A;
}

//construir tendo em conta as matrizes do matlab?
vector<double> buildVector(string input_file) {
    vector<double> b{2, 8, 9};

    return b;
}

double dotProduct(vector<double> a, vector<double> b, int size) {
	double dotProd = 0.0;
    #pragma omp parallel for reduction(+:dotProd)
	for (int i = 0; i < size; i++) {
		dotProd += (a[i] * b[i]);
	}
	return dotProd;
}

vector<double> subtractVec(vector<double> a, vector<double> b, int size) {
    vector<double> res(size);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        res[i] = a[i] - b[i];
    }
    return res;
}

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int begin, int end, int size) {
    /*Ver todos os processos que estao idle (a espera de informacao)
    escolher os processos com o id mais afastados de mim (?) e enviar-lhes iteracoes
    se nao houver nenhum processo idle, fazer so eu o trabalho
    no final, juntar o trabalho de todos os processos
    */
    vector <double> res(end - begin);
    #pragma omp parallel for
	for (int i = begin; i < end; i++) {
		res[i] = 0;
		for (int j = 0; j < size; j++) {
			res[i] += matrix[i][j] * v[j];
		}
	}
	return res;
}

/* Conjugate Gradient Method: iterative method for efficiently solving linear systems of equations: Ax=b
    Step 1: Compute gradient: g(t) = Ax(t - 1) - b
    Step 2: if g(t)^T g(t) < epsilon => return
    Step 3: Compute direction vector: d(t)
    Step 4: Compute step size: s(t)
    Step 5: Compute new aproximation: x(t) = x(t-1) + s(t)d(t)
*/
vector<double> cg(vector<vector<double>> A, vector<double> b, int size, vector<double> x, double * master, int * finalIter) {
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    vector<double> g(size); //gradient, inicializar na primeira iteraçao?
    vector<double> d = subtractVec(b, matrixVector(A, x, 0, size, size), size); //initial direction = residue
    double s; //step size

    int dest = (me == nprocs - 1 ? 0 : me + 1);
    int source = (me == 0 ? nprocs - 1 : me - 1);

    //auxiliar
    vector<double> op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    int flag=0;

    vector<double> auxGrad(size);
    vector<double> auxDir(size);

    MPI_Status status;
    MPI_Request sendGradReq;
    MPI_Request sendDirReq;
    MPI_Request sendIdleReq;

    for(int t = me; t < MAXITER; t+=nprocs) {
        if(debugParallel || debugMtr) cout << "Iteration number: " << t << ", Will be done by process " << me << endl;

        int received = 0;
        if(t != 0){
            while(received < 3) {
                MPI_Recv(&auxGrad[0], size, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                switch (status.MPI_TAG) {
                    case GRADTAG:
                        if(debugParallel) cout << "Proccess number: " << me << " Received g(t-1) from node " << source << endl;
                        #pragma omp parallel for
                        for(int i = 0; i < size; i++) {
                            g[i] = auxGrad[i];
                        }
                        received++;
                        break;
                    case DIRTAG:
                        if(debugParallel) cout << "Proccess number: " << me << " Received d(t-1) from node " << source << endl;
                        #pragma omp parallel for
                        for(int i = 0; i < size; i++) {
                            d[i] = auxGrad[i];
                        }
                        received++;
                        break;
                    case XTAG:
                        if(debugParallel) cout << "Proccess number: " << me << " Received x(t-1) from node " << source << endl;
                        #pragma omp parallel for
                        for(int i = 0; i < size; i++) {
                            x[i] = auxGrad[i];
                        }
                        received++;
                        break;
                    case ENDTAG:
                        if(status.MPI_TAG == ENDTAG && g[0] != dest) {
                            MPI_Send(&g[0], 1, MPI_DOUBLE, dest, ENDTAG, MPI_COMM_WORLD);
                            if(debugParallel) cout << "Proccess number: " << me << " Sent end tag to node " << endl;
                            return x;
                        }
                        return x;
                    default:
                        break;
                }
            }
            
        }

        if(debugParallel) cout << "Proccess number: " << me << " Received all data from node " << source << endl;

        //unico trabalho q se pode fazer antes de receber o x(t-1)
            //g(t-1)^T g(t-1)
            denom1 = dotProduct(g, g, size);

        //Ax(t-1)
        op = matrixVector(A, x, 0, size, size);

        //g(t) = Ax(t-1) - b
        g =  subtractVec(op, b, size);
        
        //g(t)^T g(t)
        num1 = dotProduct(g, g, size);
        if(debugParallel || debugMtr) cout << "Iteration number: " << t << ", num1: " << num1 << endl;
        
        if (num1 < epsilon){
            *master = me;
            *finalIter = t;
            MPI_Send(master, 1, MPI_DOUBLE, dest, ENDTAG, MPI_COMM_WORLD);
            if(debugParallel) cout << "Proccess number: " << me << " Sent end tag to node " << dest << endl;
            break;
        }
        else{
            //MPI: enviar g(t) para o proximo node asincronamente
            #pragma omp parallel for
            for(int i = 0; i < size; i++) {
                auxGrad[i] = g[i];
            }
            MPI_Isend(&auxGrad[0], size, MPI_DOUBLE, dest, GRADTAG, MPI_COMM_WORLD, &sendGradReq);
            if(debugParallel) cout << "Proccess number: " << me << " Sent g(t) to node " << dest << endl;

        } 

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        //se estivermos na iteracao 0, a direcao foi ja calculada no inicio da funcao com o residuo
        //senao, temos de receber a direcao da iteracao anterior e calcular a nova direcao
        if(t!= 0){
            #pragma omp parallel for
            for(int i = 0; i < size; i++) {
                d[i] = -g[i] + (num1/denom1) * d[i];
                if(debugMtr) 
                    cout << "Iter: " << t << "d(t)[" << i << "]: " << d[i] << endl;
                
            }
        }

        //MPI: enviar d(t) para o proximo node asincronamente
        #pragma omp parallel for
        for(int i = 0; i < size; i++){
            auxDir[i] = d[i];
        }
        MPI_Isend(&auxDir[0], size, MPI_DOUBLE, dest, DIRTAG, MPI_COMM_WORLD, &sendDirReq);

        //d(t)^T g(t)
        num2 = dotProduct(d, g, size);

        //A*d(t)
        op = matrixVector(A, d, 0, size, size);

        //d(t)^T A*d(t)
        denom2 = dotProduct(d, op, size);

        s = -num2/denom2;
        if(debugMtr){
            cout << "Iter: " << t << "num2: " << num2 << endl;
            cout << "Iter: " << t << "denom2: " << denom2 << endl;
            cout << "Iter: " << t << "s: " << s << endl;
        }

        #pragma omp parallel for
        for(int i = 0; i < size; i++) {
            x[i] = x[i] + s*d[i];
        }

        if((debugMtr || debugParallel)) {
            cout << "Iter: " << t << "x(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << "FINAL X Iter" << t <<"x(t) [" << i << "]=" << x[i] << endl;
            }
        }
        MPI_Wait(&sendGradReq, &status);
        MPI_Wait(&sendDirReq, &status);
        if(debugParallel) cout << "Proccess number: " << me << " Sending x(t) to node " << dest << endl;
        MPI_Send(&x[0], size, MPI_DOUBLE, dest, XTAG, MPI_COMM_WORLD);
        if(debugParallel) cout << "Proccess number: " << me << " Sent x(t) to node " << dest << endl;
        if(debugParallel || debugMtr) cout << "Iteration number: " << t << ", Finished" << endl;
    }
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
    vector<double> x = cg(A, b, size, b, &master, &finalIter);
    exec_time += omp_get_wtime();
    if(me == master) {
        fprintf(stderr, "%.10fs\n", exec_time);
        cout << "Master node: " << me << endl;  
        cout << "Final iteration: " << finalIter << endl;
        for(int i = 0; i < size; i++) {
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    }
    cout << "Proccess number: " << me << " Ending execution" << endl;
    MPI_Finalize();
    return 0;
}