#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include "io_ops.cpp"

using namespace std;

#define epsilon 0.00001

bool debug = false;
string input_file;

//construir tendo em conta as matrizes do matlab?
//paralelizar e distribuir por blocos?

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix(string input_file) {
    return readFile_mtx(input_file);
}

//construir tendo em conta as matrizes do matlab?

vector<double> buildVector(string inputFile, int size) {
    ifstream file(inputFile);
    string line;
    vector<double> b(size);
    /*
    int counter = 0;
    bool isDefined = false;

    while (getline(file, line)) {
        if(line[0] == '%') continue;
        vec[counter++] = stod(line);
    }
    file.close();
    */
    b[0]= 83;
    b[1]= 86;
    b[2]= 77;
    b[3]= 15;
    b[4]= 93;
    b[5]= 35;
    b[6]= 86;
    b[7]= 92;
    b[8]= 49;
    b[9]= 21;
    b[10]= 62;
    b[11]= 27;
    b[12]= 90;
    b[13]= 59;
    b[14]= 63;
    b[15]= 26;
    b[16]= 40;
    b[17]= 26;
    b[18]= 72;

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
    int maxIter = size*size;
    /* Dsitribuir as iteraçoes pelos mpi nodes */
    for(int t = 0; t < maxIter; t++) {
        cout << "============Iteration number:============" << t << endl;
        //MPI: esperar para receber valor de g(t-1) do node anterior
        if(t != 0) denom1 = dotProduct(g, g, size);
        if(debug)
            cout << "denom1: " << denom1 << endl;

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
            
        }
        cout << "s: " << s << endl;
        for(int i = 0; i < size; i++) {
            x[i] += s*d[i];
            cout << "x[" << i << "]: " << x[i] << endl;
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
    vector<double> b = buildVector("/home/cat/uni/thesis/project/conjugateGradient/Vec/Vec.txt", 19);
    int size = b.size();
    exec_time = -omp_get_wtime();
    //num max de threads
    vector<double> x = cg(A, b, size, b);
    exec_time += omp_get_wtime();
    fprintf(stdout, "%.10fs\n", exec_time);
    for(int i = 0; i < size; i++){
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    return 0;
}