#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define epsilon 0.1

//construir tendo em conta as matrizes do matlab?
vector<vector<double>> buildMatrix(/*input file?*/) {
    vector<vector<double>> A{{4, 1}, {1, 3}};
    return A;
}

//construir tendo em conta as matrizes do matlab?
vector<double> buildVector(/*input file?*/) {
    vector<double> b{ 1, 2 };
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
    //square matrix
	for (int i = 0; i < size; i++) {
		res[i] = 0;
		for (int j = 0; j < size; j++) {
			res[i] += matrix[i][j] * v[j];
		}
	}
	return res;
}

/* Conjugate Gradient Method: iterative method for efficiently solving linear systems of equations:
    Ax=b
    Step 1: Compute gradient: g(t) = Ax(t - 1) - b
    Step 2: if g(t)^T g(t) < epsilon => return
    Step 3: Compute direction vector: d(t)
    Step 4: Compute step size: s(t)
    Step 5: Compute new aproximation: x(t) = x(t-1) + s(t)d(t)
*/
vector<double> cg(vector<vector<double>> A, vector<double> b, int size, vector<double> x) {
    vector<double> g(size); //gradient, inicializar na primeira iteraçao?
    vector<double> d(size); //direction vector
    double s; //step size

    //auxiliar
    vector<double> op(size);
    double denom1 = 0;
    double denom2 = 0;
    double num1 = 0;
    double num2 = 0;
    /* Before we start, copy values of b into residual */
    for(int t = 0; t < size; t++) {
        cout << "============Iteration number:============" << t << endl;
        //g(t-1)^T g(t-1)
        if(t != 0) denom1 = dotProduct(g, g, size);

        //step 1
        //Ax(t-1)
        op = matrixVector(A, x, size);
        cout << "Ax(t-1): " << endl;
        for(int i = 0; i < size; i++) {
            cout << op[i] << endl;
        }
        //g(t) = Ax(t-1) - b
        g =  subtractVec(op, b, size);
        if(t == 0) denom1 = dotProduct(g, g, size);

        for(int i = 0; i < size && t == 0; i++) {
            d[i] = -g[i];
        }

        cout << "g(t): " << endl;
        for(int i = 0; i < size; i++) {
            cout << g[i] << endl;
        }

        //step 2
        //g(t)^T g(t)
        num1 = dotProduct(g, g, size);
        cout << "num1: " << num1 << endl;
        if (num1 < epsilon) break;

        //step 3
        //d(t) = -g(t) + (g(t)^T g(t))/(g(t-1)^T g(t-1)) * d(t-1)
        for(int i = 0; i < size && t != 0; i++) {
            cout << "-g[i]: " << -g[i] << endl;
            cout << "(num1/denom1): " << (num1/denom1) << endl;
            cout << "d[i]: " << d[i] << endl;
            d[i] = -g[i] + (num1/denom1) * d[i];
            cout << "d(t)[" << i << "]: " << d[i] << endl;
        }
        //d(t)^T g(t)
        num2 = dotProduct(d, g, size);
        cout << "num2: " << num2 << endl;
        //A*d(t)
        op = matrixVector(A, d, size);
        denom2 = dotProduct(d, op, size);
        cout << "denom2: " << denom2 << endl;
        s = -num2/denom2;
        cout << "s: " << s << endl;
        for(int i = 0; i < size; i++) {
            x[i] = x[i] + s*d[i];
        }
    }
    return x;
}

int main () {
    vector<vector<double>> A = buildMatrix();
    vector<double> b = buildVector();
    int size = b.size();
    vector<double> x0 {2, 1};
    vector<double> x = cg(A, b, size, x0);
    for(int i = 0; i < size; i++) {
        cout << "x[" << i << "]: " << x[i] << endl;
    }
    return 0;
}