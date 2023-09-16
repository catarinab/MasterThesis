#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define epsilon 1
#define MAXITER 20

#define ENDTAG 0
#define WORKTAG 1

bool debugMtr = false;
bool debugParallel = false;
string input_file;

//construir tendo em conta as matrizes do matlab?
//paralelizar e distribuir por blocos?

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

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int size) {
    vector <double> res(size);
    //TODO: chessboard decomposition com MPI
    #pragma omp parallel for
	for (int i = 0; i < size; i++) {
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
vector<double> cg(vector<vector<double>> A, vector<double> b, int size, vector<double> x) {
    int me, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    vector<double> g(size); //gradient, inicializar na primeira iteraçao?
    vector<double> d = subtractVec(b, matrixVector(A, x, size), size); //initial direction
    double s; //step size

    //auxiliar
    vector<double> op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    MPI_Status status;
    /* Distribuir as iteraçoes pelos mpi nodes */
    for(int t = 0; t < MAXITER; t++) {  
        if((me + t) % nprocs != 0) continue;
        if(debugMtr || debugParallel) cout << "============Iteration number: " << t << ", Will be done by process " << me << endl;
        //MPI: esperar para receber valor de g(t-1) do node anterior
        //MPI: esperar para receber valor de x(t-1) do node anterior
        //a seguir, podemos calcular o restantes
        if(t != 0){
            if(debugParallel) cout << "Proccess number: " << me << " Waiting to receive g(t-1) from node " << (me == 0 ? nprocs - 1 : me - 1) << endl;
            MPI_Recv(&g[0], 5, MPI_DOUBLE, (me == 0 ? nprocs - 1 : me - 1), WORKTAG, MPI_COMM_WORLD, &status);
            if(debugParallel) cout << "Proccess number: " << me << " Received g(t-1) from node " << (me == 0 ? nprocs - 1 : me - 1) << "with tag" << status.MPI_TAG << endl;
            if(status.MPI_TAG == ENDTAG && me != nprocs - 1) {
                if(debugParallel) cout << "Proccess number: " << me << " Sending g(t) to node " << me + 1 <<" and ending execution"<< endl;
                MPI_Send(&g[0], 5, MPI_DOUBLE, me + 1, ENDTAG, MPI_COMM_WORLD);
                break;
            }
            else if(status.MPI_TAG == ENDTAG && me == nprocs - 1) break;

            denom1 = dotProduct(g, g, size);

            if(debugParallel) cout << "Proccess number: " << me << " Waiting to receive x(t-1) from node " << (me == 0 ? nprocs - 1 : me - 1) << endl;
            MPI_Recv(&x[0], 5, MPI_DOUBLE, (me == 0 ? nprocs - 1 : me - 1), WORKTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //step 1
        //Ax(t-1)


        op = matrixVector(A, x, size);
        if(debugMtr){
            cout << "Ax(t-1): " << endl;
            for(int i = 0; i < size; i++) {
                cout << op[i] << endl;
            }
        }
        //g(t) = Ax(t-1) - b
        g =  subtractVec(op, b, size);

        if(debugMtr) cout << "Proccess number: " << me << " Sending g(t) to node " << (me == nprocs - 1 ? 0 : me + 1) << endl;
        

        if(debugMtr) {
            cout << "g(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << g[i] << endl;
            }
        }
        

        //step 2
        //g(t)^T g(t)
        num1 = dotProduct(g, g, size);
        if(debugMtr) cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon){
            MPI_Send(&g[0], 5, MPI_DOUBLE, (me == nprocs - 1 ? 0 : me + 1), ENDTAG, MPI_COMM_WORLD);
            break;
        }
        else MPI_Send(&g[0], 5, MPI_DOUBLE, (me == nprocs - 1 ? 0 : me + 1), WORKTAG, MPI_COMM_WORLD);

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        //se estivermos na iteracao 0, a direcao foi ja calculada no inicio da funcao com o residuo
        for(int i = 0; i < size && t != 0; i++) {
             d[i] = -g[i] + (num1/denom1) * d[i];
            if(debugMtr) {
                cout << "-g[i]: " << -g[i] << endl;
                cout << "(num1/denom1): " << (num1/denom1) << endl;
                cout << "d[i]: " << d[i] << endl;
                cout << "d(t)[" << i << "]: " << d[i] << endl;
            }
            
        }
        //d(t)^T g(t)
        num2 = dotProduct(d, g, size);

        //A*d(t)
        op = matrixVector(A, d, size);
        denom2 = dotProduct(d, op, size);
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
        if(debugParallel) cout << "Proccess number: " << me << " Sending x(t) to node " << (me == nprocs - 1 ? 0 : me + 1) << endl;
        MPI_Send(&x[0], 5, MPI_DOUBLE, (me == nprocs - 1 ? 0 : me + 1), WORKTAG, MPI_COMM_WORLD);
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
    MPI_Init(&argc, &argv);

    processInput(argc, argv);

    //dividir trabalho pelos nodes
    vector<vector<double>> A = buildMatrix(input_file);
    vector<double> b = buildVector(input_file);
    int size = 3;
    //num max de threads
    omp_set_num_threads(size);
    //initial guess: b
    vector<double> x = cg(A, b, size, b);
    for(int i = 0; i < size; i++) {
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    MPI_Finalize();
    return 0;
}