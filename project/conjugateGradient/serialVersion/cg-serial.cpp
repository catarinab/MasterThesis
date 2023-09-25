#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define epsilon 0.000001
#define MAXITER 20

bool debug = false;
string input_file;

//construir tendo em conta as matrizes do matlab?
//paralelizar e distribuir por blocos?

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

double dotProduct(vector<double> a, vector<double> b, int size) {
	double dotProd = 0.0;
	for (int i = 0; i < size; i++) {
		dotProd += (a[i] * b[i]);
	}
	return dotProd;
}

vector<double> subtractVec(vector<double> a, vector<double> b, int size) {
    vector<double> res(size);
    for (int i = 0; i < size; i++) {
        res[i] = a[i] - b[i];
    }
    cout << "res: " << endl;
    for(int i = 0; i < size; i++) {
        cout << res[i] << endl;
    }
    return res;
}

vector<double> matrixVector(vector<vector<double>> matrix, vector<double> v, int size) {
    vector <double> res(size);
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
    vector<double> g(size); //gradient, inicializar na primeira iteraçao?
    vector<double> d = subtractVec(b, matrixVector(A, x, size), size); //initial direction
    double s; //step size

    //auxiliar
    vector<double> op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    /* Dsitribuir as iteraçoes pelos mpi nodes */
    for(int t = 0; t < MAXITER; t++) {
        cout << "============Iteration number:============" << t << endl;
        //MPI: esperar para receber valor de g(t-1) do node anterior
        if(t != 0) denom1 = dotProduct(g, g, size);

        //step 1
        //Ax(t-1)
        //MPI: esperar para receber valor de x(t-1) do node anterior
        op = matrixVector(A, x, size);
        if(debug){
            cout << "Ax(t-1): " << endl;
            for(int i = 0; i < size; i++) {
                cout << op[i] << endl;
            }
        }
        //g(t) = Ax(t-1) - b
        g =  subtractVec(op, b, size);
        if(debug) {
            cout << "g(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << g[i] << endl;
            }
        }
        

        //step 2
        //g(t)^T g(t)
        num1 = dotProduct(g, g, size);
        if(debug) cout << "num1: " << num1 << endl;
        
        if (num1 < epsilon) break;

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        //se estivermos na iteracao 0, a direcao foi ja calculada no inicio da funcao com o residuo
        for(int i = 0; i < size && t != 0; i++) {
             d[i] = -g[i] + (num1/denom1) * d[i];
            if(debug) {
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
        if(debug){
            cout << "num2: " << num2 << endl;
            cout << "denom2: " << denom2 << endl;
            cout << "s: " << s << endl;
        }
        for(int i = 0; i < size; i++) {
            x[i] = x[i] + s*d[i];
        }
        if(debug) {
            cout << "x(t): " << endl;
            for(int i = 0; i < size; i++) {
                cout << x[i] << endl;
            }
        }
    }
    return x;
}

void processInput(int argc, char* argv[]) {
    for(int i = 0; i < argc; i++) {
        if(string(argv[i]) == "-d") {
            debug = true;
        }
        if(string(argv[i]) == "-f") {
            input_file = string(argv[i+1]);
        }
    }
}

int main (int argc, char* argv[]) {
    double exec_time;       /* execution time */
    processInput(argc, argv);

    vector<vector<double>> A = buildMatrix(input_file);
    vector<double> b = buildVector(input_file);
    int size = b.size();
    exec_time = -omp_get_wtime();
    //num max de threads
    vector<double> x0 {2, 1};
    vector<double> x = cg(A, b, size, b);
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.10fs\n", exec_time);
    for(int i = 0; i < size; i++) {
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    return 0;
}