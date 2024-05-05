#include <vector>
#include <iostream>
#include <complex>
#include <bits/stdc++.h>
#include <mkl.h>
#include <omp.h>

#include "headers/dense_matrix.hpp"
#include "headers/utils.hpp"
#include "headers/calculate-MLF.hpp"
#include "headers/schur-blocking.hpp"
#include "headers/Evaluate-Single-ML-parallel.hpp"

using namespace std;

/*
Algorithm based on the paper "Computing the matrix Mittag–Leffler function with applications to fractional calculus"
by Roberto Garrappa and Marina Popolizio
*/

//auxiliary variables for matrix multiplication
complex<double> alphaMult = {1, 0};
complex<double> betaMult = {0, 0};

string folderMLF("0-1/");

void printMatrixFile(const string& fileName, const string& name, complex<double> * matrix, int size) {
    ofstream myFile;
    myFile.open(folderMLF + fileName);
    myFile << name << " =[ " ;
    for(int i = 0; i < size; i++ ) {
        for(int j = 0; j < size; j++){
            myFile << scientific << std::setprecision (15) << matrix[i + j * size].real() << "+"
                   << matrix[i + j * size].imag() << "i";
            if(j != size - 1)
                myFile <<",";
        }
        if(i != size -1)
            myFile << ";";
    }
    myFile << "];" << endl;
    myFile.close();
}

void printMatrixFile(const string& fileName, const string& name, double * matrix, int size) {
    ofstream myFile;
    myFile.open(folderMLF + fileName);
    myFile << name << " =[ " ;
    for(int i = 0; i < size; i++ ) {
        for(int j = 0; j < size; j++){
            myFile << scientific << std::setprecision (15) << matrix[i + j * size];
            if(j != size - 1)
                myFile <<",";
        }
        if(i != size -1)
            myFile << ";";
    }
    myFile << "];" << endl;
    myFile.close();
}

void printMatrix(const string& name, complex<double> * matrix, int rowMin, int rowMax, int colMin, int colMax) {
    int size = colMax - colMin + 1;
    for (int i = 0; i <= (rowMax - rowMin); i++) {
        for (int j = 0; j <= (colMax - colMin); j++) {
            cout << name <<": " << scientific << setprecision(16) <<
                 matrix[i + j * size].real() << " + " << matrix[i + j * size].imag() << "i" << endl;
        }
    }
    cout << endl;
}

void getSubMatrix(complex<double> ** subMatrix, complex<double> * matrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; i++) {
        for (int j = 0; j < subCols; j++) {
            (*subMatrix)[i + j * subRows] = matrix[(rowMin + i) + (colMin + j) * size];
        }
    }
}

void printMainMatrix(const string& name,complex<double> * matrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    for (int i = rowMin; i <= rowMax; i++) {
        for (int j = colMin; j <= colMax; j++) {
            cout << name <<": " << scientific << setprecision(15) <<
                 matrix[i + j * size].real() << " + " << matrix[i + j * size].imag() << "i" << endl;
        }
    }
}

void setMainMatrix(complex<double> ** A, complex<double> * subMatrix, int i, int elSize, int size) {
    for(int row = i ; row < i + elSize; row++)
        for(int col = i ; col < i + elSize; col++) {
            if(row <= col)
                (*A)[row + col * size] = subMatrix[(row - i) + (col - i) * elSize];
        }
}

void setMainMatrix(complex<double> ** A, complex<double> * subMatrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; i++) {
        for (int j = 0; j < subCols; j++) {
            (*A)[(rowMin + i) + (colMin + j) * size] = subMatrix[i + j * subRows];
        }
    }
}

/*temp = T(i,j)*(F(i,i) - F(j,j)) + F(i,k)*T(k,j) - T(i,k)*F(k,j);
F(i,j) = temp/(T(i,i)-T(j,j));*/
complex<double> single_eq(complex<double> * fA, complex<double> * T, int i, int j, int size) {
    complex<double> result = {0.0, 0.0};
    // tij * (fii - fjj)
    complex<double> numerator = T[i + j * size] * (fA[i + i * size] - fA[j + j * size]);

    int kMin = i + 1;
    int kMax = j - 1;
    int kSize = kMax - kMin + 1;
    if(kSize > 0 && (i + 1) <= (j - 1)) {

        auto * Fik = (complex<double> *) calloc((1 * kSize), sizeof(complex<double>));
        getSubMatrix(&Fik, fA, i, i, kMin, kMax, size);

        auto * Fkj = (complex<double> *) calloc((kSize * 1), sizeof(complex<double>));
        getSubMatrix(&Fkj, fA, kMin, kMax, j, j, size);

        auto * Tkj = (complex<double> *) calloc((kSize * 1), sizeof(complex<double>));
        getSubMatrix(&Tkj, T, kMin, kMax, j, j, size);

        auto * Tik = (complex<double> *) calloc((1 * kSize), sizeof(complex<double>));
        getSubMatrix(&Tik, T, i, i, kMin, kMax, size);

        complex<double> temp;
        //numerator = numerator +  Fik * Tkj
        cblas_zdotu_sub(kSize, Fik, 1, Tkj, 1, &temp);
        numerator += temp;

        //numerator = numerator - Tik * Fkj
        cblas_zdotu_sub(kSize, Tik, 1, Fkj, 1, &temp);
        numerator -= temp;
    }

    //tii - tjj
    complex<double> denominator = T[i + i * size] - T[j + j * size];
    //(tij * (fii - fjj) + fik * tkj - tik * fkj) / (tii - tjj)
    result = numerator / denominator;
    return result;
}


complex<double> * evaluateBlock(complex<double> * T, double alpha, double beta,
                                vector<int> element, int tSize) {
    //default values
    double tol = EPS16;
    int maxTerms = 250;
    int maxDeriv = 1;

    int i = element[0];
    int elSize = (int) element.size();

    int elThreads = omp_get_max_threads() > elSize ? elSize : omp_get_max_threads();

    auto * M = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * auxMatrixLpck = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * P = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * F = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * F_old = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * F_aux = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));

    auto * auxMatrix = (double *) calloc(elSize * elSize, sizeof(double));
    auto *  ones = (double *) calloc(elSize, sizeof(double));

    complex<double> lambda = {0, 0};
    for(int row = i ; row < i + elSize; row++){
        lambda += T[row + row * tSize];
    }

    //lambda = trace(T)/n
    lambda /= (double) elSize;

    //cout << fixed << scientific << setprecision(15) << "lambda: " << lambda << endl;

    //M = T - lambda*I
    //auxMatrix = I - abs(triu(T,1));
    #pragma omp parallel for num_threads(elThreads)
    for(int j = 0; j < elSize; j++){
        for(int k = 0; k < elSize; k++){
            if(j == k) {
                M[j + k * elSize] = T[(i + j) + (i + k) * tSize] - lambda;
                auxMatrix[j + k * elSize] = 1.0;
            }
            else {
                if (k > j)
                    auxMatrix[j + k * elSize] = - abs(T[(i + j) + (i + k) * tSize]);
                M[j + k * elSize] = T[(i + j) + (i + k) * tSize];
            }
        }
        ones[j] = 1.0;
    }

    LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'U', elSize, 1, auxMatrix,
                   elSize, ones, elSize);

    memcpy(P, M, elSize * elSize * sizeof(complex<double>));

    complex<double> f = evaluateSingle(lambda, alpha, beta, 0);

    //cout << "f = " << f << endl;
    double mu = 0;
    // F = f*I
    // mu = infNorm(ones)
    #pragma omp parallel for reduction(max:mu) num_threads(elThreads)
    for(int ii = 0; ii < elSize; ii++) {
        F[ii + ii * elSize] = f;
        mu = max(mu, abs(ones[ii]));
    }

    /*cout << "Initial F:" << endl;
    printMatrix("F", F, 0, elSize - 1, 0, elSize - 1);*/

    for(int k = 1; k <= maxTerms; k++){
        double norm_F_F_old;
        double norm_F_old;
        double norm_F;
        f = evaluateSingle(lambda, alpha, beta, k);
        //cout << scientific << setprecision(16) << "f at iteration " << k << " is: " << f << endl;

        //F_old = F
        memcpy(F_old, F, elSize * elSize * sizeof(complex<double>));

        //F = F + P*f
        //F_aux = F - F_old
        #pragma omp parallel for
        for(int ii = 0; ii < elSize; ii++){
            for(int j = 0; j < elSize; j++){
                F[ii + j * elSize] = F[ii + j * elSize] +  P[ii + j * elSize] * f;
                F_aux[ii + j * elSize] = F[ii + j * elSize] - F_old[ii + j * elSize];
            }
        }

        /*cout << "F at iteration " << k << ":" << endl;
        printMatrix("F", F, 0, elSize - 1, 0, elSize - 1);*/
        norm_F = LAPACKE_zlange(LAPACK_COL_MAJOR, 'I', elSize, elSize,
                                reinterpret_cast<const MKL_Complex16 *>(F), elSize);
        norm_F_old = LAPACKE_zlange(LAPACK_COL_MAJOR, 'I', elSize, elSize,
                                    reinterpret_cast<const MKL_Complex16 *>(F_old), elSize);
        norm_F_F_old = LAPACKE_zlange(LAPACK_COL_MAJOR, 'I', elSize, elSize,
                                      reinterpret_cast<const MKL_Complex16 *>(F_aux), elSize);


        //P = P*N/(k+1);
        complex<double> alphaP = 1.0/(k+1.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    elSize, elSize, elSize, &alphaP, P, elSize, M,
                    elSize, &betaMult, auxMatrixLpck, elSize);

        memcpy(P, auxMatrixLpck, elSize * elSize * sizeof(complex<double>));

        /*cout << "P at iteration " << k << ":" << endl;
        printMatrix("P", P, 0, elSize - 1, 0, elSize - 1);*/
        double relDiff = norm_F_F_old/(tol + norm_F_old);

        if(relDiff <= tol) {
            vector<double> fDerivMax = vector<double>(maxTerms + elSize - 1);

            int nrThreads = omp_get_max_threads() > k + elSize - maxDeriv ? k + elSize - maxDeriv : omp_get_max_threads();

            int restThreads = floor((omp_get_max_threads() - nrThreads) / nrThreads);

            #pragma omp parallel for schedule(dynamic) num_threads(nrThreads)
            for(int j = maxDeriv; j < k + elSize ; j++){
                //evaluate values for diagonal of the block
                //calculate w
                //calculate delta
                for(int jj = element[0]; jj < element[0] + elSize; jj++){
                    complex<double> res = evaluateSingle(T[jj + jj * tSize], alpha, beta, j, restThreads);
                    fDerivMax[j]  = max(fDerivMax[j] , abs(res));
                }
                //cout << scientific << setPrecision(15) << "fDerivMax[" << j << "] = " << fDerivMax[j] << endl;
            }

            maxDeriv = k+elSize;

            double omega = 0;
            for(int j = 0; j < elSize; j++){
                omega = max(omega, fDerivMax[k + j]/factorial(j));
            }

            double norm_P = LAPACKE_zlange(LAPACK_COL_MAJOR, 'I', elSize, elSize,
                                           reinterpret_cast<const MKL_Complex16 *>(P), elSize);


            if(norm_P*mu*omega <= tol*norm_F){
                /*cout << "found solution:" << endl;
                printMatrix("F", F, 0, elSize - 1,
                            0, elSize - 1);*/
                free(P);
                free(F_old);
                free(F_aux);
                free(ones);
                free(auxMatrix);
                free(auxMatrixLpck);
                free(M);
                return F;
            }
        }
    }

    cout << "DIDNT CONVERGE" << endl;
    free(P);
    free(F_old);
    free(F_aux);
    free(ones);
    free(auxMatrix);
    free(auxMatrixLpck);
    free(M);
    return F;
}

dense_matrix calculate_MLF(double * A, double alpha, double beta, int size) {

    double exec_time;

    auto * T = (complex<double> *) calloc(size * size, sizeof(complex<double>));
    auto * U = (complex<double> *) calloc(size * size, sizeof(complex<double>));
    //important: only the necessary fields in fA are filled, the other ones *must* be assigned to 0 -> use calloc
    auto * fA = (complex<double> *) calloc(size * size, sizeof(complex<double>));

    vector<vector<int>> ind = schurDecomposition(A, &T, &U, size);

    exec_time = -omp_get_wtime();

    //evaluate diagonal entries (blocks or single entries)
    for(int col = 0; col < ind.size(); col++){
        vector<int> j = ind[col];
        int elSize = (int) j.size();
        int elLine = j[0];
        if(elSize == 1) {
            fA[elLine + elLine * size] = evaluateSingle(T[elLine + elLine * size], alpha, beta, 0);
            /*cout  << fixed << scientific << setPrecision(15) << "val," << T[elLine + elLine * size].real() << " + " <<
            T[elLine + elLine * size].imag() << "i,single,i," << elLine + 1 << ",val,"
                 << fA[elLine + elLine * size].real() << " + " << fA[elLine + elLine * size].imag() << "i" << endl;*/
        }
        else {
            complex<double> * F = evaluateBlock(T, alpha, beta, j, size);
            setMainMatrix(&fA, F, elLine, elSize, size);
            free(F);
        }

        //Parlett recursion
        for (int row = col - 1; row >= 0; row--) {
            vector<int> i = ind[row];
            if(i.size() == 1 && j.size() == 1) {
                fA[i[0] + j[0] * size] = single_eq(fA, T, i[0], j[0], size);
                /*cout << "singleeq,i," << i[0] + 1
                     << ",j," << j[0] + 1 << ",val," << fA[i[0] * size + j[0]].real() << " + "
                     << fA[i[0] * size + j[0]].imag() << "i" << endl;*/
            }
            else{
                int jSize = (int) j.size();
                int iSize = (int) i.size();
                int jMin = j[0];
                int jMax = j[jSize - 1];
                int iMin = i[0];
                int iMax = i[iSize - 1];

                int kMin = ind[row + 1][0];
                int kMax = ind[col - 1][ind[col - 1].size() - 1];
                int kSize = kMax - kMin + 1;

                bool kMult = kSize > 0 && (row + 1) <= (col - 1);

                /*cout << "iMin," << iMin << ",iMax," << iMax <<
                     ",jMin," << jMin << ",jMax," << jMax << endl;*/
                /*if(kMult)
                    cout << "kMin," << kMin << ",kMax," << kMax << endl;*/

                //triangular
                auto *Fii = (complex<double> *) calloc((iSize * iSize), sizeof(complex<double>));
                getSubMatrix(&Fii, fA, iMin, iMax, iMin, iMax, size);
                //printMatrix("Fii", Fii, iMin, iMax, iMin, iMax);

                //triangular
                auto *Fjj = (complex<double> *) calloc((jSize * jSize), sizeof(complex<double>));
                getSubMatrix(&Fjj, fA, jMin, jMax, jMin, jMax, size);
                //printMatrix("Fjj", Fjj, jMin, jMax, jMin, jMax);

                auto *Tij = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));
                getSubMatrix(&Tij, T, iMin, iMax, jMin, jMax, size);

                //printMatrix("Tij", Tij, iMin, iMax, jMin, jMax);

                //triangular
                auto *Tii = (complex<double> *) calloc((iSize * iSize), sizeof(complex<double>));
                getSubMatrix(&Tii, T, iMin, iMax, iMin, iMax, size);
                //printMatrix("Tii", Tii, iMin, iMax, iMin, iMax);

                //triangular
                auto *Tjj = (complex<double> *) calloc((jSize * jSize), sizeof(complex<double>));
                getSubMatrix(&Tjj, T, jMin, jMax, jMin, jMax, size);
                //printMatrix("Tjj", Tjj, jMin, jMax, jMin, jMax);

                auto * Fii_Tij = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));

                auto * Tij_Fjj = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));

                auto * Fik_Tkj = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));
                auto * Tik_Fkj = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));
                auto * result = (complex<double> *) calloc((iSize * jSize), sizeof(complex<double>));

                //Fii * Tij
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                             iSize, jSize, iSize, &alphaMult, Fii, iSize,
                            Tij, iSize, &betaMult, Fii_Tij, iSize);
                //Tij * Fjj
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                             iSize, jSize, jSize, &alphaMult, Tij, iSize,
                            Fjj, jSize, &betaMult, Tij_Fjj, iSize);

                /*printMatrix("Fii * Tij", Fii_Tij, iMin, iMax, jMin, jMax);
                printMatrix("Tij * Fjj", Tij_Fjj, iMin, iMax, jMin, jMax);*/

                if(kMult){

                    auto *Fik = (complex<double> *) calloc((iSize * kSize), sizeof(complex<double>));
                    getSubMatrix(&Fik, fA, iMin, iMax, kMin, kMax, size);
                    //printMatrix("Fik", Fik, iMin, iMax, kMin, kMax);

                    auto *Fkj = (complex<double> *) calloc((kSize * jSize), sizeof(complex<double>));
                    getSubMatrix(&Fkj, fA, kMin, kMax, jMin, jMax, size);
                    //printMatrix("Fkj", Fkj, kMin, kMax, jMin, jMax);

                    auto *Tkj = (complex<double> *) calloc((kSize * jSize), sizeof(complex<double>));
                    getSubMatrix(&Tkj, T, kMin, kMax, jMin, jMax, size);

                    auto *Tik = (complex<double> *) calloc((iSize * kSize), sizeof(complex<double>));
                    getSubMatrix(&Tik, T, iMin, iMax, kMin, kMax, size);

                    //Fik*Tkj
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                 iSize, jSize, kSize, &alphaMult, Fik, iSize,
                                Tkj, kSize, &betaMult, Fik_Tkj, iSize);

                    //printMatrix("Fik*Tkj", aux3, 0, iMax - iMin, 0, jMax - jMin);


                    //Tik * Fkj
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                 iSize, jSize, kSize, &alphaMult, Tik, iSize,
                                Fkj, kSize, &betaMult, Tik_Fkj, iSize);

                    //printMatrix("Tik * Fkj", aux4, 0, iMax - iMin, 0, jMax - jMin);

                    free(Fik);
                    free(Fkj);
                    free(Tkj);
                    free(Tik);
                }
                //Fij
                for(int ii = 0; ii < iSize; ii++){
                    for(int jj = 0; jj < jSize; jj++){
                        result[ii + jj * iSize] = Fii_Tij[ii + jj * iSize] - Tij_Fjj[ii + jj * iSize] +
                                                     Fik_Tkj[ii + jj * iSize] - Tik_Fkj[ii + jj * iSize];
                    }
                }

                //printMatrix("rhs", result, 0, iMax - iMin, 0, jMax - jMin);

                //Sylvester equation to find Fij
                //Tii * Fij - Fij * Tjj = aux
                double scale = 1.0;

                LAPACKE_ztrsyl(LAPACK_COL_MAJOR, 'N', 'N', -1, iSize,
                               jSize, reinterpret_cast<const MKL_Complex16 *>(Tii), iSize,
                               reinterpret_cast<const MKL_Complex16 *>(Tjj), jSize,
                               reinterpret_cast<MKL_Complex16 *>(result), iSize, &scale);



                if(scale != 1) {
                    cout << "scale = " <<  scale << endl;
                    for (int ii = 0; ii < iSize; ii++) {
                        for (int jj = 0; jj < jSize; jj++) {
                            result[ii + jj * iSize] /= scale;
                        }
                    }
                }

                setMainMatrix(&fA, result, iMin, iMax, jMin, jMax, size);

                //printMainMatrix("finalResult", fA, iMin, iMax, jMin, jMax, size);

                free(Fii);
                free(Fjj);
                free(Tij);
                free(Tii);
                free(Tjj);
                free(Fii_Tij);
                free(Tij_Fjj);
                free(Fik_Tkj);
                free(Tik_Fkj);
                free(result);
            }
        }
    }

    exec_time += omp_get_wtime();

    /*cout << "Execution time: " << exec_time << endl;*/

    auto * temp = (complex<double> *) calloc(size * size, sizeof(complex<double>));

    //return to A
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, &alphaMult, U, size, fA, size, &betaMult, temp, size);


    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                size, size, size, &alphaMult, temp, size, U, size, &betaMult, T, size);

    dense_matrix res = dense_matrix(size, size);

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            res.setValue(i, j, T[i + j * size].real());
        }
    }

    free(temp);
    free(fA);
    free(U);
    free(T);

    return res;
}

