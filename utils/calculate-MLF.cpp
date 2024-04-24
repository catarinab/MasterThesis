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
#include "headers/Evaluate-Single-ML.hpp"


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
            myFile << scientific << std::setprecision (15) << matrix[i * size + j].real() << "+"
                   << matrix[i * size + j].imag() << "i";
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
            myFile << scientific << std::setprecision (15) << matrix[i * size + j];
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
                 matrix[i * size + j].real() << " + " << matrix[i * size + j].imag() << "i" << endl;
        }
    }
    cout << endl;
}

void getSubMatrix(complex<double> ** subMatrix, complex<double> * matrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; i++) {
        for (int j = 0; j < subCols; j++) {
            (*subMatrix)[i * subCols + j] = matrix[(rowMin + i) * size + (colMin + j)];
        }
    }
}

void printMainMatrix(const string& name,complex<double> * matrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    for (int i = rowMin; i <= rowMax; i++) {
        for (int j = colMin; j <= colMax; j++) {
            cout << name <<": " << scientific << setprecision(15) <<
                 matrix[i * size + j].real() << " + " << matrix[i * size + j].imag() << "i" << endl;
        }
    }
}

void setMainMatrix(complex<double> ** A, complex<double> * subMatrix, int i, int elSize, int size) {
    for(int row = i ; row < i + elSize; row++)
        for(int col = i ; col < i + elSize; col++) {
            if(row <= col)
                (*A)[row * size + col] = subMatrix[(row - i) * elSize + col - i];
        }
}

void setMainMatrix(complex<double> ** A, complex<double> * subMatrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; i++) {
        for (int j = 0; j < subCols; j++) {
            (*A)[(rowMin + i) * size + (colMin + j)] = subMatrix[i * subCols + j];
        }
    }
}

/*temp = T(i,j)*(F(i,i) - F(j,j)) + F(i,k)*T(k,j) - T(i,k)*F(k,j);
F(i,j) = temp/(T(i,i)-T(j,j));*/
complex<double> single_eq(complex<double> * fA, complex<double> * T, int i, int j, int size) {
    complex<double> result = {0.0, 0.0};
    // tij * (fii - fjj)
    complex<double> numerator = T[i * size + j] * (fA[i * size + i] - fA[j * size + j]);

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
    complex<double> denominator = T[i * size + i] - T[j * size + j];
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

    auto * M = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * auxMatrixLpck = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));
    auto * P = (complex<double> *) calloc(elSize * elSize,  sizeof(complex<double>));
    auto * F = (complex<double> *) calloc(elSize * elSize,  sizeof(complex<double>));
    auto * F_old = (complex<double> *) calloc(elSize * elSize,  sizeof(complex<double>));
    auto * F_aux = (complex<double> *) calloc(elSize * elSize, sizeof(complex<double>));

    auto * auxMatrix = (double *) calloc(elSize * elSize, sizeof(double));
    auto *  ones = (double *) calloc(elSize, sizeof(double));

    complex<double> lambda = {0, 0};
    for(int row = i ; row < i + elSize; row++){
        lambda += T[row * tSize + row];
    }

    //lambda = trace(T)/n
    lambda /= (double) elSize;

    //cout << fixed << scientific << setprecision(15) << "lambda: " << lambda << endl;

    //M = T - lambda*I
    //auxMatrix = I - abs(triu(T,1));
    #pragma omp parallel for
    for(int j = 0; j < elSize; j++){
        for(int k = 0; k < elSize; k++){
            if(j == k) {
                M[j * elSize + k] = T[(i + j) * tSize + (i + k)] - lambda;
                auxMatrix[j * elSize + k] = 1.0;
            }
            else {
                if (k > j)
                    auxMatrix[j * elSize + k] = - abs(T[(i + j) * tSize + (i + k)]);
                M[j * elSize + k] = T[(i + j) * tSize + (i + k)];
            }
        }
        ones[j] = 1.0;
    }

    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'U', elSize, 1, auxMatrix,
                   elSize, ones, 1);

    memcpy(P, M, elSize * elSize * sizeof(complex<double>));

    complex<double> f = evaluateSingle(lambda, alpha, beta, 0);

    //cout << "f = " << f << endl;
    double mu = 0;
    // F = f*I
    // mu = infNorm(ones)
    #pragma omp parallel for reduction(max:mu)
    for(int ii = 0; ii < elSize; ii++) {
        F[ii * elSize + ii] = f;
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
                F[ii * elSize + j] = F[ii * elSize + j] +  P[ii * elSize + j] * f;
                F_aux[ii * elSize + j] = F[ii * elSize + j] - F_old[ii * elSize + j];
            }
        }

        /*cout << "F at iteration " << k << ":" << endl;
        printMatrix("F", F, 0, elSize - 1, 0, elSize - 1);*/
        norm_F = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'I', elSize, elSize,
                                reinterpret_cast<const MKL_Complex16 *>(F), elSize);
        norm_F_old = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'I', elSize, elSize,
                                    reinterpret_cast<const MKL_Complex16 *>(F_old), elSize);
        norm_F_F_old = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'I', elSize, elSize,
                                      reinterpret_cast<const MKL_Complex16 *>(F_aux), elSize);


        //P = P*N/(k+1);
        complex<double> alphaP = 1.0/(k+1.0);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    elSize, elSize, elSize, &alphaP, P, elSize, M,
                    elSize, &betaMult, auxMatrixLpck, elSize);

        memcpy(P, auxMatrixLpck, elSize * elSize * sizeof(complex<double>));

        /*cout << "P at iteration " << k << ":" << endl;
        printMatrix("P", P, 0, elSize - 1, 0, elSize - 1);*/
        double relDiff = norm_F_F_old/(tol + norm_F_old);

        if(relDiff <= tol) {
            vector<double> fDerivMax = vector<double>(maxTerms + elSize - 1);

            int nr_threads = omp_get_max_threads() > k + elSize - maxDeriv ? k + elSize - maxDeriv : omp_get_max_threads();

            #pragma omp parallel for schedule(dynamic) num_threads(nr_threads)
            for(int j = maxDeriv; j < k + elSize ; j++){
                //evaluate values for diagonal of the block
                //calculate w
                //calculate delta
                for(int jj = element[0]; jj < element[0] + elSize; jj++){
                    complex<double> res = evaluateSingle(T[jj * tSize + jj], alpha, beta, j);
                    fDerivMax[j]  = max(fDerivMax[j] , abs(res));
                }
                //cout << scientific << setPrecision(15) << "fDerivMax[" << j << "] = " << fDerivMax[j] << endl;
            }

            maxDeriv = k+elSize;

            double omega = 0;
            for(int j = 0; j < elSize; j++){
                omega = max(omega, fDerivMax[k + j]/factorial(j));
            }

            double norm_P = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'I', elSize, elSize,
                                           reinterpret_cast<const MKL_Complex16 *>(P), elSize);

            /*cout << scientific << setPrecision(15) << "omega = " << omega << endl;
            cout << scientific << setPrecision(15) << "mu = " << mu << endl;
            cout << scientific << setPrecision(15) << "norm_P = " << norm_P << endl;
            cout << scientific << setPrecision(15) << "norm_F = " << norm_F << endl;*/

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
    free(F);
    return nullptr;
}

dense_matrix calculate_MLF(double * A, double alpha, double beta, int size) {

    double exec_time = 0;

    auto * T = (complex<double> *) calloc(size * size, sizeof(complex<double>));
    auto * U = (complex<double> *) calloc(size * size, sizeof(complex<double>));
    //important: only the necessary fields in fA are filled, the other ones *must* be assigned to 0 -> use calloc
    auto * fA = (complex<double> *) calloc(size * size, sizeof(complex<double>));

    vector<vector<int>> ind = schurDecomposition(A, &T, &U, size);

    exec_time = -omp_get_wtime();

    /*printMatrixFile("T-16.txt", "T", T, size);
    printMatrixFile("U-16.txt", "U", U, size);*/

    /*ofstream myFile;
    myFile.open(folderMLF + "ind.txt");
    myFile << "ind = cell(" << ind.size() << ", 1);" << endl;
    for (size_t i = 0; i < ind.size(); ++i) {
        myFile << "ind{" << i + 1 << "} = [";
        for (size_t j = 0; j < ind[i].size(); ++j) {
            myFile << ind[i][j] + 1;
            if (j != ind[i].size() - 1) {
                myFile << ", ";
            }
        }
        myFile << "];" << endl;
    }
    myFile.close();*/

    /*for (const auto & i : ind) {
       cout << i.size() << endl;
    }*/

    //evaluate diagonal entries (blocks or single entries)
    for(int col = 0; col < ind.size(); col++){
        vector<int> j = ind[col];
        int elSize = (int) j.size();
        int elLine = j[0];
        if(elSize == 1) {
            fA[elLine * size + elLine] = evaluateSingle(T[elLine * size + elLine], alpha, beta, 0);
            /*cout  << fixed << scientific << setPrecision(15) << "val," << T[elLine * size + elLine].real() << " + " <<
            T[elLine * size + elLine].imag() << "i,single,i," << elLine + 1 << ",val,"
                 << fA[elLine * size + elLine].real() << " + " << fA[elLine * size + elLine].imag() << "i" << endl;*/
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
                fA[i[0] * size + j[0]] = single_eq(fA, T, i[0], j[0], size);
                /*cout << "singleeq,i," << i[0] + 1
                     << ",j," << j[0] + 1 << ",val," << fA[i[0] * size + j[0]].real() << " + "
                     << fA[i[0] * size + j[0]].imag() << "i" << endl;*/
            }
            else{
                int jSize = (int) j.size();
                int iSize = (int) i.size();
                int jMin = j[0];
                int jMax = j[j.size() - 1];
                int iMin = i[0];
                int iMax = i[i.size() - 1];

                int kMin = ind[row + 1][0];
                int kMax = ind[col - 1][ind[col - 1].size() - 1];
                int kSize = kMax - kMin + 1;

                bool kMult = kSize > 0 && (row + 1) <= (col - 1);

                /*cout << "iMin," << iMin << ",iMax," << iMax <<
                     ",jMin," << jMin << ",jMax," << jMax << endl;*/
                /*if(kMult)
                    cout << "kMin," << kMin << ",kMax," << kMax << endl;*/

                //triangular
                auto *Fii = (complex<double> *) calloc((i.size() * i.size()), sizeof(complex<double>));
                getSubMatrix(&Fii, fA, iMin, iMax, iMin, iMax, size);
                //printMatrix("Fii", Fii, iMin, iMax, iMin, iMax);

                //triangular
                auto *Fjj = (complex<double> *) calloc((j.size() * j.size()), sizeof(complex<double>));
                getSubMatrix(&Fjj, fA, jMin, jMax, jMin, jMax, size);
                //printMatrix("Fjj", Fjj, jMin, jMax, jMin, jMax);

                auto *Tij = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));
                getSubMatrix(&Tij, T, iMin, iMax, jMin, jMax, size);

                //printMatrix("Tij", Tij, iMin, iMax, jMin, jMax);

                //triangular
                auto *Tii = (complex<double> *) calloc((i.size() * i.size()), sizeof(complex<double>));
                getSubMatrix(&Tii, T, iMin, iMax, iMin, iMax, size);
                //printMatrix("Tii", Tii, iMin, iMax, iMin, iMax);

                //triangular
                auto *Tjj = (complex<double> *) calloc((j.size() * j.size()), sizeof(complex<double>));
                getSubMatrix(&Tjj, T, jMin, jMax, jMin, jMax, size);
                //printMatrix("Tjj", Tjj, jMin, jMax, jMin, jMax);

                auto * Fii_Tij = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));

                auto * Tij_Fjj = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));

                auto * Fik_Tkj = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));
                auto * Tik_Fkj = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));
                auto * result = (complex<double> *) calloc((i.size() * j.size()), sizeof(complex<double>));

                //Fii * Tij
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int) i.size(), (int) j.size(), (int) iSize, &alphaMult, Fii, (int) iSize,
                            Tij, (int) jSize, &betaMult, Fii_Tij, (int) jSize);
                //Tij * Fjj
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int) i.size(), (int) j.size(), (int) jSize, &alphaMult, Tij, (int) jSize,
                            Fjj, (int) jSize, &betaMult, Tij_Fjj, (int) jSize);

                /*printMatrix("Fii * Tij", Fii_Tij, iMin, iMax, jMin, jMax);
                printMatrix("Tij * Fjj", Tij_Fjj, iMin, iMax, jMin, jMax);*/

                if(kMult){

                    auto *Fik = (complex<double> *) calloc((i.size() * kSize), sizeof(complex<double>));
                    getSubMatrix(&Fik, fA, iMin, iMax, kMin, kMax, size);
                    //printMatrix("Fik", Fik, iMin, iMax, kMin, kMax);

                    auto *Fkj = (complex<double> *) calloc((kSize * j.size()), sizeof(complex<double>));
                    getSubMatrix(&Fkj, fA, kMin, kMax, jMin, jMax, size);
                    //printMatrix("Fkj", Fkj, kMin, kMax, jMin, jMax);

                    auto *Tkj = (complex<double> *) calloc((kSize * j.size()), sizeof(complex<double>));
                    getSubMatrix(&Tkj, T, kMin, kMax, jMin, jMax, size);

                    auto *Tik = (complex<double> *) calloc((i.size() * kSize), sizeof(complex<double>));
                    getSubMatrix(&Tik, T, iMin, iMax, kMin, kMax, size);

                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) i.size(), (int) j.size(), kSize, &alphaMult, Fik, kSize,
                                Tkj, (int) j.size(), &betaMult, Fik_Tkj, (int) j.size());

                    //printMatrix("Fik*Tkj", aux3, 0, iMax - iMin, 0, jMax - jMin);


                    //Tik * Fkj
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) i.size(), (int) j.size(), kSize, &alphaMult, Tik, kSize,
                                Fkj, (int) j.size(), &betaMult, Tik_Fkj, (int) j.size());

                    //printMatrix("Tik * Fkj", aux4, 0, iMax - iMin, 0, jMax - jMin);

                    free(Fik);
                    free(Fkj);
                    free(Tkj);
                    free(Tik);
                }
                for(int ii = 0; ii < i.size(); ii++){
                    for(int jj = 0; jj < j.size(); jj++){
                        result[ii * j.size() + jj] = Fii_Tij[ii * j.size() + jj] - Tij_Fjj[ii * j.size() + jj] +
                                                     Fik_Tkj[ii * j.size() + jj] - Tik_Fkj[ii * j.size() + jj];
                    }
                }

                //printMatrix("rhs", result, 0, iMax - iMin, 0, jMax - jMin);

                //Sylvester equation to find Fij
                //Tii * Fij - Fij * Tjj = aux
                double scale = 1.0;

                LAPACKE_ztrsyl(LAPACK_ROW_MAJOR, 'N', 'N', -1, (int) i.size(),
                               (int) j.size(), reinterpret_cast<const MKL_Complex16 *>(Tii), (int) i.size(),
                               reinterpret_cast<const MKL_Complex16 *>(Tjj), (int) j.size(),
                               reinterpret_cast<MKL_Complex16 *>(result), (int) j.size(), &scale);



                if(scale != 1) {
                    cout << "scale = " <<  scale << endl;
                    for (int ii = 0; ii < i.size(); ii++) {
                        for (int jj = 0; jj < j.size(); jj++) {
                            result[ii * j.size() + jj] /= scale;
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
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, &alphaMult, U, size, fA, size, &betaMult, temp, size);


    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                size, size, size, &alphaMult, temp, size, U, size, &betaMult, T, size);

    dense_matrix res = dense_matrix(size, size);

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            res.setValue(i, j, T[i * size + j].real());
        }
    }

    free(temp);
    free(fA);
    free(U);
    free(T);

    return res;
}

