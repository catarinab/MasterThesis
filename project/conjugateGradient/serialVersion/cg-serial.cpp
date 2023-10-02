#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include "io_ops.cpp"

using namespace std;

#define epsilon 0.001
#define MAXITER 100

bool debug = false;
string input_file;

//construir tendo em conta as matrizes do matlab?
//paralelizar e distribuir por blocos?

//positive definite matrix !! (symmetric matrix whose every eigenvalue is positive.)
vector<vector<double>> buildMatrix(string input_file) {
    return readFile_mtx(input_file);
}

//construir tendo em conta as matrizes do matlab?

vector<double> buildVector(string input_file) {
    vector<double> b(200);
    b[0]=83;
b[1]=86;
b[2]=77;
b[3]=15;
b[4]=93;
b[5]=35;
b[6]=86;
b[7]=92;
b[8]=49;
b[9]=21;
b[10]=62;
b[11]=27;
b[12]=90;
b[13]=59;
b[14]=63;
b[15]=26;
b[16]=40;
b[17]=26;
b[18]=72;
b[19]=36;
b[20]=11;
b[21]=68;
b[22]=67;
b[23]=29;
b[24]=82;
b[25]=30;
b[26]=62;
b[27]=23;
b[28]=67;
b[29]=35;
b[30]=29;
b[31]=2;
b[32]=22;
b[33]=58;
b[34]=69;
b[35]=67;
b[36]=93;
b[37]=56;
b[38]=11;
b[39]=42;
b[40]=29;
b[41]=73;
b[42]=21;
b[43]=19;
b[44]=84;
b[45]=37;
b[46]=98;
b[47]=24;
b[48]=15;
b[49]=70;
b[50]=13;
b[51]=26;
b[52]=91;
b[53]=80;
b[54]=56;
b[55]=73;
b[56]=62;
b[57]=70;
b[58]=96;
b[59]=81;
b[60]=5;
b[61]=25;
b[62]=84;
b[63]=27;
b[64]=36;
b[65]=5;
b[66]=46;
b[67]=29;
b[68]=13;
b[69]=57;
b[70]=24;
b[71]=95;
b[72]=82;
b[73]=45;
b[74]=14;
b[75]=67;
b[76]=34;
b[77]=64;
b[78]=43;
b[79]=50;
b[80]=87;
b[81]=8;
b[82]=76;
b[83]=78;
b[84]=88;
b[85]=84;
b[86]=3;
b[87]=51;
b[88]=54;
b[89]=99;
b[90]=32;
b[91]=60;
b[92]=76;
b[93]=68;
b[94]=39;
b[95]=12;
b[96]=26;
b[97]=86;
b[98]=94;
b[99]=39;
b[100]=95;
b[101]=70;
b[102]=34;
b[103]=78;
b[104]=67;
b[105]=1;
b[106]=97;
b[107]=2;
b[108]=17;
b[109]=92;
b[110]=52;
b[111]=56;
b[112]=1;
b[113]=80;
b[114]=86;
b[115]=41;
b[116]=65;
b[117]=89;
b[118]=44;
b[119]=19;
b[120]=40;
b[121]=29;
b[122]=31;
b[123]=17;
b[124]=97;
b[125]=71;
b[126]=81;
b[127]=75;
b[128]=9;
b[129]=27;
b[130]=67;
b[131]=56;
b[132]=97;
b[133]=53;
b[134]=86;
b[135]=65;
b[136]=6;
b[137]=83;
b[138]=19;
b[139]=24;
b[140]=28;
b[141]=71;
b[142]=32;
b[143]=29;
b[144]=3;
b[145]=19;
b[146]=70;
b[147]=68;
b[148]=8;
b[149]=15;
b[150]=40;
b[151]=49;
b[152]=96;
b[153]=23;
b[154]=18;
b[155]=45;
b[156]=46;
b[157]=51;
b[158]=21;
b[159]=55;
b[160]=79;
b[161]=88;
b[162]=64;
b[163]=28;
b[164]=41;
b[165]=50;
b[166]=93;
b[167]=0;
b[168]=34;
b[169]=64;
b[170]=24;
b[171]=14;
b[172]=87;
b[173]=56;
b[174]=43;
b[175]=91;
b[176]=27;
b[177]=65;
b[178]=59;
b[179]=36;
b[180]=32;
b[181]=51;
b[182]=37;
b[183]=28;
b[184]=75;
b[185]=7;
b[186]=74;
b[187]=21;
b[188]=58;
b[189]=95;
b[190]=29;
b[191]=37;
b[192]=35;
b[193]=93;
b[194]=18;
b[195]=28;
b[196]=43;
b[197]=11;
b[198]=28;
b[199]=29;
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
    /* Dsitribuir as iteraçoes pelos mpi nodes */
    for(int t = 0; t < MAXITER; t++) {
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
    vector<double> x = cg(A, b, size, b);
    exec_time += omp_get_wtime();
    fprintf(stdout, "%.10fs\n", exec_time);
    double sum = 0;
    for(int i = 0; i < size; i++) sum += x[i];
    cout << "Sum: " << sum << endl;
    return 0;
}