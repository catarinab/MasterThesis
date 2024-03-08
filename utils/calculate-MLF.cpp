#include <cmath>
#include <vector>
#include <iostream>
#include <complex>
#include <functional>
#include <bits/stdc++.h>

#include "headers/dense_matrix.hpp"
#include "headers/utils.hpp"
#include "headers/mtx_ops_mkl.hpp"
#include "headers/calculate-MLF.hpp"
#include "headers/schur-blocking.hpp"
#include "headers/MLF-LTI.hpp"

/*
Algorithm based on the paper "Computing the matrix Mittag–Leffler function with applications to fractional calculus" 
by Roberto Garrappa and Marina Popolizio
*/

int Jmax;

lpck_c alphaMult = {1.0, 0.0};
lpck_c betaMult = {0.0, 0.0};

void printMatrix(const string& name, lpck_c * matrix, int size) {
    cout << name << " =[ " ;
    for(int i = 0; i < size; i++ ) {
        for(int j = 0; j < size; j++){
            cout << matrix[i * size + j].real << " + " << matrix[i * size + j].imag << "i ";
            if(j != size - 1)
                cout <<",";
        }
        if(i != size -1)
            cout << ";";
    }
    cout << "];" << endl;

}

void printMatrix(const string& name, lpck_c * matrix, int rowMin, int rowMax, int colMin, int colMax) {
    int size = colMax - colMin + 1;
    cout << name << " =[ " ;
    for (int i = 0; i <= (rowMax - rowMin); ++i) {
        for (int j = 0; j <= (colMax - colMin); ++j) {
            cout << matrix[i * size + j].real << " + " << matrix[i * size + j].imag << "i ";
            if (j != colMax)
                cout << ",";
        }
        if(i != rowMax)
            cout << ";";
    }
    cout << "];" << endl;
}

void getSubMatrix(lpck_c ** subMatrix, lpck_c * matrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; ++i) {
        for (int j = 0; j < subCols; ++j) {
            (*subMatrix)[i * subCols + j] = matrix[(rowMin + i) * size + (colMin + j)];
        }
    }
}

void setMainMatrix(lpck_c ** A, lpck_c * subMatrix, int i, int elSize, int size) {
    for(int row = i ; row < i + elSize; row++)
        for(int col = i ; col < i + elSize; col++) {
            (*A)[row * size + col] = subMatrix[(row - i) * elSize + col - i];
        }
}


void setMainMatrix(lpck_c ** A, lpck_c * subMatrix, int rowMin, int rowMax, int colMin, int colMax, int size) {
    int subRows = rowMax - rowMin + 1;
    int subCols = colMax - colMin + 1;
    for (int i = 0; i < subRows; ++i) {
        for (int j = 0; j < subCols; ++j) {
            (*A)[(rowMin + i) * size + (colMin + j)] = subMatrix[i * subCols + j];
        }
    }
}

lpck_c single_eq(lpck_c * fA, lpck_c * T, int i, int j, int size) {
    lpck_c result = {0.0, 0.0};
    lpck_c numerator1 = lpck_z_sub(fA[i * size + i], fA[j * size + j]);
    lpck_c denominator1 = lpck_z_sub(T[i * size + i], T[j * size + j]);
    lpck_c div1 = lpck_z_div(numerator1, denominator1);
    lpck_c currSum = lpck_z_mult(T[i * size + j], div1);
    result = lpck_z_sum(result, currSum);
    for(int k = i + 1; k < j; k++){
        lpck_c result1 = lpck_z_mult(fA[i * size + k], T[k * size + j]);
        lpck_c result2 = lpck_z_mult(T[i * size + k], fA[k * size + j]);
        lpck_c numerator = lpck_z_sub(result1, result2);
        lpck_c denominator = lpck_z_sub(T[i * size + i], T[j * size + j]);
        lpck_c div = lpck_z_div(numerator, denominator);
        result = lpck_z_sum(result, div);
    }
    return result;
}

lpck_c evaluateSingle(lpck_c lambda, double alpha, double beta, int k) {
    complex<double> tVal = complex<double>(lambda.real, lambda.imag);
    complex<double> result;
    lpck_c resultLapack;
    bool accept = false;

    //target accuracy threshold
    double tau = 1.0e-14;

    double numerator = tgamma(alpha*Jmax + beta);
    double denominator = falling_factorial(Jmax, k);
    double bound = pow((tau * numerator/denominator), 1/(Jmax - k));

    if(abs(tVal) < bound){
        result = series_expansion(tVal, alpha, beta, &accept, k);
        resultLapack = {result.real(), result.imag()};
    }

    if(!accept){
        if(abs(tVal) <= EPS)
            resultLapack = {1/tgamma(beta), 0};
        else {
            result = LTI(tVal, alpha, beta, k);
            resultLapack = {result.real(), result.imag()};
        }
    }

    return resultLapack;
}

lpck_c * evaluateBlock(lpck_c * T, double alpha, double beta, vector<int> element, int tSize) {
    //default values
    double tol = EPS16;
    int maxTerms = 250;
    int maxDeriv = 1;

    int i = element[0];
    int elSize = (int) element.size();
    //cout << "elSize: " << elSize << endl;

    auto * block = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * M = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * auxMatrixLpck = (lpck_c *) calloc(elSize * elSize, sizeof(lpck_c));
    auto * identity = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * upperBlock = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * P = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * F = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * F_old = (lpck_c *) calloc(elSize * elSize,  sizeof(lpck_c));
    auto * F_aux = (lpck_c *) calloc(elSize * elSize, sizeof(lpck_c));
    lpck_c aux;

    auto * auxMatrix = (lpck_c *) calloc(elSize * elSize, sizeof(lpck_c));
    auto *  ones = (lpck_c *) calloc(elSize , sizeof(lpck_c));

    vector<double> fDerivMax = vector<double>(maxTerms);
    complex<double> diagSum = {0, 0};

    cout << "block = [";
    for(int row = i ; row < i + elSize; row++){
        for(int col = i ; col < i + elSize; col++){
            lpck_c tVal = T[row * tSize + col];
            if(row == col){
                identity[(row - i) * elSize + col - i] = {1, 0};
                diagSum += complex<double>{tVal.real, tVal.imag};
            }
            if(col > row)
                upperBlock[(row - i) * elSize + col - i] = tVal;
            block[(row - i) * elSize + col - i]  = tVal;
            cout << block[(row - i) * elSize + col - i];
            if(col != i + elSize - 1)
                cout << ",";
        }
        if(row != i + elSize - 1)
            cout <<";";
    }
    cout << "]" << endl;

    lpck_c lambda = {diagSum.real()/elSize, diagSum.imag()/elSize};

    for(int j = 0; j < elSize; j++){
        for(int k = 0; k < elSize; k++){
            if(j == k)
                M[j * elSize + k] = lpck_z_sub(T[(i + j) * tSize + (i + k)], lambda);
            else
                M[j * elSize + k] = T[(i + j) * tSize + (i + k)];
        }
    }



    for(int ii = 0; ii < elSize; ii++){
        for(int j = 0; j < elSize; j++){
            auxMatrix[ii * elSize + j] = lpck_z_sub(identity[ii * elSize + j], upperBlock[ii * elSize + j]);
        }
        ones[ii] = {1,0};
    }

    LAPACKE_ztrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', elSize, 1, auxMatrix,
                   elSize, ones, 1);

    double mu = 0;
    for(int ii = 0; ii < elSize; ii++)
        mu = max(mu, lpck_abs(ones[ii]));


    memcpy(P, M, elSize * elSize * sizeof(lpck_c));

    lpck_c f = evaluateSingle(lambda, alpha, beta, 0);

    for(int ii = 0; ii < elSize; ii++){
        for(int j = 0; j < elSize; j++){
            if(ii == j)
                F[ii * elSize + j] = {f.real, 0};
            else
                F[ii * elSize + j] = {0, 0};
        }
    }

    for(int k = 1; k <= maxTerms; k++){
        double norm_F_F_old = 0;
        double norm_F_old = 0;
        double norm_F = 0;
        f = evaluateSingle(lambda, alpha, beta, k);

        //F_old = F
        memcpy(F_old, F, elSize * elSize * sizeof(lpck_c));

        //F = F + P*f; certo
        //cout << " F at iteration " << k << " is " << endl;
        for(int ii = 0; ii < elSize; ii++){
            double row_norm_F_F_old = 0;
            double row_norm_F_old = 0;
            double row_norm_F = 0;
            for(int j = 0; j < elSize; j++){
                lpck_c fP = lpck_z_mult(P[ii * elSize + j], f);
                F[ii * elSize + j] = lpck_z_sum(F[ii * elSize + j], fP);
                //cout << "F, col: " << ii << " row: " << j << " val: " << F[ii * elSize + j] << endl;
                row_norm_F_F_old += lpck_abs(lpck_z_sub(F[ii * elSize + j], F_old[ii * elSize + j]));
                row_norm_F_old += lpck_abs(F_old[ii * elSize + j]);
                row_norm_F += lpck_abs(F[ii * elSize + j]);
            }
            norm_F_F_old = max(norm_F_F_old, row_norm_F_F_old);
            norm_F = max(norm_F, row_norm_F);
            norm_F_old = max(norm_F_old, row_norm_F_old);
        }

        //cout << "norm_F_F_old " << norm_F_F_old << endl;
        //cout << "norm_F_old " << norm_F_old << endl;
        //cout << "norm_F " << norm_F << endl;


        //P = P*M/(k+1);
        //cout << "P at iteration " << k << " is " << endl;
        lpck_c alphaP = {(double) (1.0/(k+1.0)), 0.0};
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    elSize, elSize, elSize, &alphaP, P, elSize, M,
                    elSize, &betaMult,auxMatrixLpck, elSize);
        memcpy(P, auxMatrixLpck, elSize * elSize * sizeof(lpck_c));

        double relDiff = norm_F_F_old/(tol + norm_F_old);

        if(relDiff <= tol) {
            double delta = 0;
            cout << "iteration " << k << " got into the if" << endl;
            for(int r = maxDeriv; r < k + elSize ; r++){
                //evaluate values for diagonal of the block
                //calculate w
                //calculate delta
                for(int col = 0; col < elSize; col++){
                    aux = evaluateSingle(T[col*elSize + col], alpha, beta, r);
                    fDerivMax[r-maxDeriv] = max(fDerivMax[r], lpck_abs(aux));
                }
                delta = max(delta, abs(fDerivMax[r-maxDeriv]/factorial(r-maxDeriv)));
            }

            maxDeriv = k+elSize;

            double norm_P = 0;
            for(int ii = 0; ii < elSize; ii++){
                double rowNorm = 0;
                for(int j = 0; j < elSize; j++){
                    rowNorm += lpck_abs(P[ii * elSize + j]);
                    //cout << "P, col: " << ii << " row: " << j << " val: " << P[ii * elSize + j] << endl;
                }
                norm_P = max(norm_P, rowNorm);
            }

            if(mu*delta*norm_P <= tol*norm_F){
                cout << "Found result:" << endl;
                for(int ii = 0; ii < elSize; ii++ ) {
                    for(int j = 0; j < elSize; j++){
                        cout << "row: " << ii << " col: " << j << " val: " << F[ii * elSize + j] << endl;
                    }
                }
                free(P);
                free(F_old);
                free(F_aux);
                free(ones);
                free(upperBlock);
                free(identity);
                free(auxMatrix);
                free(auxMatrixLpck);
                free(M);
                free(block);
                return F;
            }
        }
    }

    cout << "DIDNT CONVERGE" << endl;
    free(P);
    free(F_old);
    free(F_aux);
    free(ones);
    free(upperBlock);
    free(identity);
    free(auxMatrix);
    free(auxMatrixLpck);
    free(M);
    free(block);
    return F;
}


complex<double> series_expansion(complex<double> A, double alpha, double beta, bool * accept,
                                    int kd) {

    vector<double> sum_args = vector<double>(Jmax+1-kd);
    complex<double> result = complex<double>(0,0);
    complex<double> error = complex<double>(0,0);

    if (abs(A) < EPS){
        *accept = true;
        return 1/tgamma(beta);
    }

    for(int j = kd; j <= Jmax; j++){
        double denominator = tgamma(alpha*j + beta);
        double numerator = falling_factorial(j, kd);
        complex<double> sumVal = numerator/denominator * pow(A, j-kd);
        result += sumVal;
        sum_args[j - kd] = abs(sumVal);
    }

    //cout << "result: " << result << endl;

    sort(sum_args.begin(), sum_args.end());

    for(int jj = kd; jj <= Jmax; jj++){
        error += abs(sum_args[jj])* (Jmax - jj);
    }

    error += Jmax * sum_args[0];

    error = error * EPS;

    *accept = abs(error - Jmax * sum_args[0]) <= abs(error);

    return result;
}


dense_matrix calculate_MLF(dense_matrix A, double alpha, double beta, int size) {

    //maximal argument for the gamma function
    double max_gamma_arg = 171.624;
    //upper bound for the number of terms in the series expansion
    Jmax = floor((max_gamma_arg - beta)/alpha);

    auto * T = (lpck_c *) malloc(size * size * sizeof(lpck_c));
    auto * U = (lpck_c *) malloc(size * size * sizeof(lpck_c));

    //important: only the necessary fields in fA are filled, the other ones *must* be assigned to 0 -> use calloc
    auto * fA = (lpck_c *) calloc(size * size, sizeof(lpck_c));

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            T[i * size + j].real = A.getValue(i, j);
        }
    }


    vector<vector<int>> ind = schurDecomposition(&T, &U, size);

    //evaluate diagonal entries (blocks or single entries)
    for(int col = 0; col < ind.size(); col++){
        vector<int> j = ind[col];
        int elSize = (int) j.size();
        int elLine = j[0];
        if(elSize == 1) {
            fA[elLine * size + elLine] = evaluateSingle(T[elLine * size + elLine], alpha, beta, 0);
        }
        else {
            lpck_c * F = evaluateBlock(T, alpha, beta, j, size);
            setMainMatrix(&fA, F, elLine, elSize, size);
            free(F);
        }

        //Parlett recursion
        for (int row = col - 1; row >= 0; row--) {
            vector<int> i = ind[row];
            if(i.size() == 1 && j.size() == 1) {
                fA[i[0] * size + j[0]] = single_eq(fA, T, i[0], j[0], size);
            }
            else{
                int jMin = j[0];
                int jMax = j[j.size() - 1];
                int iMin = i[0];
                int iMax = i[i.size() - 1];

                /*cout << "jMin: " << jMin << "jMax: " << jMax << endl;
                cout << "iMin: " << iMin << "iMax: " << iMax << endl;*/

                auto *Fii = (lpck_c *) calloc((i.size() * i.size()), sizeof(lpck_c));
                getSubMatrix(&Fii, fA, iMin, iMax, iMin, iMax, size);
                //printMatrix("Fii", Fii, iMin, iMax, iMin, iMax);

                auto *Fjj = (lpck_c *) calloc((j.size() * j.size()), sizeof(lpck_c));
                getSubMatrix(&Fjj, fA, jMin, jMax, jMin, jMax, size);
                //printMatrix("Fjj", Fjj, jMin, jMax, jMin, jMax);

                auto *Tij = (lpck_c *) calloc((i.size() * j.size()), sizeof(lpck_c));
                getSubMatrix(&Tij, T, iMin, iMax, jMin, jMax, size);
                //printMatrix("Tij", Tij, iMin, iMax, jMin, jMax);

                auto *Tii = (lpck_c *) calloc((i.size() * i.size()), sizeof(lpck_c));
                getSubMatrix(&Tii, T, iMin, iMax, iMin, iMax, size);
                //printMatrix("Tii", Tii, iMin, iMax, iMin, iMax);

                auto *Tjj = (lpck_c *) calloc((j.size() * j.size()), sizeof(lpck_c));
                getSubMatrix(&Tjj, T, jMin, jMax, jMin, jMax, size);
                //printMatrix("Tjj", Tjj, jMin, jMax, jMin, jMax);

                auto *aux = (lpck_c *) calloc((i.size() * j.size()), sizeof(lpck_c));
                auto *aux1 = (lpck_c *) calloc((i.size() * j.size()), sizeof(lpck_c));
                auto *aux2 = (lpck_c *) calloc((i.size() * j.size()), sizeof(lpck_c));

                // Fii * Tij
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int) i.size(), (int) j.size(), (int) i.size(), &alphaMult, Fii, (int) i.size(),
                            Tij, (int) j.size(), &betaMult, aux1,(int) j.size());

                //printMatrix("Fii * Tij", aux1, iMin, iMax, jMin, jMax);

                // Tij * Fjj
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int) i.size(), (int) j.size(), (int) j.size(), &alphaMult, Tij,
                            (int) j.size(), Fjj, (int) j.size(), &betaMult, aux2,
                            (int) j.size());

                //printMatrix("Tij * Fjj", aux2, iMin, iMax, jMin, jMax);

                //Fii * Tij - Tij * Fjj
                for (int ii = 0; ii < i.size(); ii++) {
                    for (int jj = 0; jj < j.size(); jj++) {
                        aux[ii * j.size() + jj] = lpck_z_sub(aux1[ii * j.size() + jj], aux2[ii * j.size() + jj]);
                        /*cout <<"aux[" << ii << "," << jj <<"] = " << aux[ii * j.size() + jj].real << "+" <<
                        aux[ii * j.size() + jj].imag << "i" << endl;*/
                    }
                }

                int kMin = ind[row+1][0];
                int kMax = ind[col - 1][ind[col - 1].size() - 1];
                int kSize = kMax - kMin + 1;

                if ((row + 1 <= col - 1) && kSize > 0) {
                    //cout << "kMin: " << kMin << "kMax: " << kMax << endl;
                    auto *Fik = (lpck_c *) calloc((i.size() * kSize), sizeof(lpck_c));
                    getSubMatrix(&Fik, fA, iMin, iMax, kMin, kMax, size);
                    //printMatrix("Fik", Fik, iMin, iMax, kMin, kMax);

                    auto *Fkj = (lpck_c *) calloc((kSize * j.size()), sizeof(lpck_c));
                    getSubMatrix(&Fkj, fA, kMin, kMax, jMin, jMax, size);
                    //printMatrix("Fkj", Fkj, kMin, kMax, jMin, jMax);

                    auto *Tkj = (lpck_c *) calloc((kSize * j.size()), sizeof(lpck_c));
                    getSubMatrix(&Tkj, T, kMin, kMax, jMin, jMax, size);
                    //printMatrix("Tkj", Tkj, kMin, kMax, jMin, jMax);

                    auto *Tik = (lpck_c *) calloc((i.size() * kSize), sizeof(lpck_c));
                    getSubMatrix(&Tik, T, iMin, iMax, kMin, kMax, size);
                    //printMatrix("Tik", Tik, iMin, iMax, kMin, kMax);

                    //Fik * Tkj
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) i.size(), (int) j.size(), kSize, &alphaMult, Fik, kSize,
                                Tkj, (int) j.size(), &betaMult, aux1,(int) j.size());

                    //printMatrix("Fik * Tkj", aux1, iMin, iMax, jMin, jMax);


                    //Tik * Fkj
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) i.size(), (int) j.size(), kSize, &alphaMult, Tik, kSize,
                                Fkj, (int) j.size(), &betaMult, aux2,(int) j.size());

                    //printMatrix("Tik * Fkj", aux2, iMin, iMax, jMin, jMax);

                    //aux = aux + Fik * Tkj - Tik * Fkj
                    for (int ii = 0; ii < i.size(); ii++) {
                        for (int jj = 0; jj < j.size(); jj++) {
                            lpck_c temp = lpck_z_sub(aux1[ii * j.size() + jj], aux2[ii * j.size() + jj]);
                            aux[ii * j.size() + jj] = lpck_z_sum(aux[ii * j.size() + jj], temp);
                            /*cout <<"aux[" << ii << "," << jj <<"] = " << aux[ii * j.size() + jj] << endl; */
                        }
                    }

                    free(Fik);
                    free(Fkj);
                    free(Tkj);
                    free(Tik);
                }

                //Sylvester equation to find Fij
                //Tii * Fij - Fij * Tjj = aux
                double scale;
                LAPACKE_ztrsyl(LAPACK_ROW_MAJOR, 'N', 'N', -1, (int) i.size(),
                               (int) j.size(),Tii, (int) i.size(), Tjj, (int) j.size(), aux,
                               (int) j.size(), &scale);

                if(scale != 1) {
                    cout << "scale: " << scale << endl;
                    for (int ii = 0; ii < i.size(); ii++) {
                        for (int jj = 0; jj < j.size(); jj++) {
                            aux[ii * j.size() + jj] = lpck_z_div(aux[ii * j.size() + jj], {scale, 0});
                        }
                    }
                }


                setMainMatrix(&fA, aux, iMin, iMax, jMin, jMax, size);

                free(Fii);
                free(Fjj);
                free(Tij);
                free(Tii);
                free(Tjj);
                free(aux);
                free(aux1);
                free(aux2);
            }
        }
    }

    auto * A1 = (lpck_c *) calloc(size * size , sizeof(lpck_c));

    auto * A2 = (lpck_c *) calloc(size * size , sizeof(lpck_c));


    //return to A
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
     size, size, size, &alphaMult, U, size, fA, size, &betaMult, A1, size);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
     size, size, size, &alphaMult, A1, size, U, size, &betaMult, A2, size);

    //A2 contains result of A = U * A * U^H

    dense_matrix E = lapackeToDenseMatrix(A2, size, size);

    free(A1);
    free(A2);
    free(fA);
    free(T);
    free(U);

    return E;
}

