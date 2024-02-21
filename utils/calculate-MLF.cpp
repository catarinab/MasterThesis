#include "headers/dense_matrix.hpp"
#include "headers/utils.hpp"
#include "headers/mtx_ops_mkl.hpp"
#include <Eigen/Eigenvalues> 
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>    // std::reverse
#include <limits>
#include <complex>
#include <functional>
#include <bits/stdc++.h>

/*
Algorithm based on the paper "Computing the matrix Mittag–Leffler function with applications to fractional calculus" 
by Roberto Garrappa and Marina Popolizio
*/

using namespace Eigen;

struct s_star_struct {
    complex<double> s_star;
    double phi_star = 0;

    bool operator==(const s_star_struct &other) const {
        return s_star == other.s_star && phi_star == other.phi_star;
    }
};

struct sums {
    complex<double> sum;
    double abs;

    bool operator>(const sums &other) const {
        return abs > other.abs;
    }
};

double factorial(int n) {
    double result = 1;
    for(int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}


complex<double> series_expansion(complex<double> A, double alpha, double beta, double Jmax, bool * accept) {

    vector<double> sum_args = vector<double>(Jmax+1);
    complex<double> result = complex<double>(0,0);
    complex<double> error = complex<double>(0,0);

    if (abs(A) < EPS){
        *accept = true;
        return 1/tgamma(beta);
    }
    
    else for(int j = 0; j <= Jmax; j++){
        double denominator = tgamma(alpha*j + beta);
        double numerator = 1; //falling factorial com k = 0
        complex<double> sumVal = numerator/denominator * pow(A, j); //igual
        result += sumVal;
        sum_args[j] = abs(sumVal);
    }

    cout << "result: " << result << endl;
    
    sort(sum_args.begin(), sum_args.end());

    for(int jj = 1; jj <= Jmax; jj++){
        error += abs(sum_args[jj])* (Jmax - jj);
    }

    error += Jmax * sum_args[0];

    error = error * EPS;

    *accept = abs(error - Jmax * sum_args[0]) <= abs(error);
    
    return result;
}


complex<double> residues(double alpha, double beta, vector<s_star_struct> s_star_vec) {
    if(s_star_vec.size() == 0)
        return complex<double>(0,0);
    
    valarray<complex<double>> res = valarray<complex<double>>(s_star_vec.size());
    for(int i = 0; i < res.size(); i++) {
        //cout << "s_star_vec[i].s_star " << s_star_vec[i].s_star << endl;
        complex<double> r1 = 1/alpha * exp(s_star_vec[i].s_star);
        complex<double> r2 = pow(s_star_vec[i].s_star, 1-beta);
        res[i] = r1 * r2;
    }
    return res.sum();
}


void optimal_param_RB(double phi, double phi1, double p, double q, double log_epsilon, double * mu, double * N, double * h) {
    cout << "RB" << endl;
    double log_eps = -36.043653389117154;
    double fac = 1.01; 
    double f_max = exp(log_epsilon - log_eps);
    double f_min;
    
    double sqrt_phi = sqrt(phi);
    double threshold = 2* sqrt(log_epsilon - log_eps);
    double sqrt_phi1 = min(sqrt(phi1), threshold - sqrt_phi);
    double sqrt_phibar;
    double sqrt_phibar1;
    double f_bar;
    int admRegion = 0;

    if(p < 1e-14 && q < 1e-14) {
        sqrt_phibar = sqrt_phi;
        sqrt_phibar1 = sqrt_phi1;
        admRegion = 1;
    }
    else if(p < 1e-14 && q >= 1e-14) {
        sqrt_phibar = sqrt_phi;
        if(sqrt_phi > 0)
            f_min = pow(fac*(sqrt_phi/(sqrt_phi1 - sqrt_phi)), q);
        else
            f_min = fac;
        if(f_min < f_max){
            f_bar = f_min + f_min/f_max * (f_max - f_min);
            double fq = pow(f_bar, -1/q);
            sqrt_phibar1 = (2*sqrt_phi1-fq*sqrt_phi)/(2+fq);
            admRegion = 1;
        }
        else
            admRegion = 0;
    }
    else if (p >= 1e-14 && q < 1e-14) {
        sqrt_phibar1 = sqrt_phi1;
        f_min = pow(fac*(sqrt_phi1/(sqrt_phi1 - sqrt_phi)), p);

        if(f_min < f_max){
            f_bar = f_min + f_min/f_max * (f_max - f_min);
            double fp = pow(f_bar, -1/p);
            sqrt_phibar = (2*sqrt_phi-fp*sqrt_phi1)/(2+fp);
            admRegion = 1;
        }
        else
            admRegion = 0;
    }
    else {
        f_min = pow(fac*(sqrt_phi+sqrt_phi1)/(sqrt_phi1 - sqrt_phi), max(p, q));
            
        if(f_min < f_max){
            f_min = max(f_min, 1.5);
            f_bar = f_min + f_min/f_max * (f_max - f_min);
            double fp = pow(f_bar, -1/p);
            double fq = pow(f_bar, -1/q);
            double w = -sqrt_phi1/log_epsilon;
            double den = 2+w - (1+w)*fq + fq;
            sqrt_phibar = ((2+w+fq)*sqrt_phi + fq*sqrt_phi1)/den;
            sqrt_phibar1 = (-(1+w)*fq*sqrt_phi + (2+w - (1+w)*fp)*sqrt_phi1)/den;
            admRegion = 1;
        }
        else
            admRegion = 0;
    }

    if(admRegion) {
        log_epsilon = log_epsilon - log(f_bar);
        double w = -pow(sqrt_phibar1, 2)/log_epsilon;
        *mu = pow((((1+w)*sqrt_phibar + sqrt_phibar1)/(2+w)),2);
        *h = -2*M_PI/log_epsilon*(sqrt_phibar1-sqrt_phibar) /((1+w)*sqrt_phibar + sqrt_phibar1) ;
        *N = ceil(sqrt(1-log_epsilon/ *mu)/ *h) ;
        
    }
    else {
        *mu = 0;
        *N = std::numeric_limits<double>::infinity();
        *h = 0;
    }

    
}


void optimal_param_RU(double phi, double p, double log_epsilon, double * mu, double * N, double * h) {
    //cout << "RU" << endl;
    double sqrt_phi = sqrt(phi);
    double phi_bar;
    double sqrt_phibar;
    int admRegion = 0;
    double sq_mu;
    double A;
    
    if(phi > 0)
        phi_bar =phi*1.01;
    else
        phi_bar = 0.01;

    sqrt_phibar = sqrt(phi_bar);

    int f_min = 1 ; int f_max = 10 ; int f_tar = 5 ;

    bool stop = false;

    while(!stop) {
        double log_eps_phi = log_epsilon/phi_bar;
        *N = ceil(phi_bar/M_PI*(1 - 3*log_eps_phi/2 + sqrt(1-2*log_eps_phi)));
        A = M_PI * *N / phi_bar;
        sq_mu = sqrt_phibar*abs(4-A)/abs(7-sqrt(1+12*A));
        double fbar = pow(((sqrt_phibar-sqrt_phi)/sq_mu), -p) ;
        stop = (p < 1.0e-14) || (f_min < fbar && fbar < f_max);
        if(!stop) {
            sqrt_phibar = pow(f_tar, -1/p) * sq_mu + sqrt_phi ;
            phi_bar = pow(sqrt_phibar,2) ;
        }
    }
    *mu = pow(sq_mu,2) ;
    *h = (-3*A - 2 + 2*sqrt(1+12*A))/(4-A)/ *N;

    double log_eps = log(EPS16); 
    double threshold = log_epsilon - log_eps;
    double Q;

    if(*mu > threshold) {
        if(abs(p) < 1e-14)
            Q = 0;
        else
            Q = pow(f_tar, -1/p) * sqrt(*mu);

        phi_bar = pow(Q + sqrt(phi),2);
        if(phi_bar < threshold) {
            double w = sqrt(log_eps/(log_eps-log_epsilon)) ;
            double u = sqrt(-phi_bar/log_eps) ;
            *mu = threshold;
            *N = ceil(w*log_epsilon/2/M_PI/(u*w-1));
            *h = sqrt(log_eps/(log_eps - log_epsilon))/ *N ;
        }
        else{
            *N = std::numeric_limits<double>::infinity(); 
            *h = 0;
        }
    }
    
}   


complex<double> LTI(complex<double> lambda, double alpha, double beta) {
    double result = 0;
    int k = 0;

    double log_epsilon = log(1e-15);
    double log_eps = log(EPS);

    //calculate phase angle of lambda
    double theta = arg(lambda);
    
    double kmin = ceil(-alpha/2 - theta/2/M_PI);
    double kmax = floor(alpha/2 - theta/2/M_PI);

    vector <s_star_struct> temp_s_star_vec = vector<s_star_struct>(kmax - kmin + 1);
    vector <s_star_struct> s_star_vec = vector<s_star_struct>(kmax - kmin + 1);

    //calculate values of s_star for valid values of j (kk)
    double abs_lambda = pow(abs(lambda), 1/alpha);
    for(int kk = kmin; kk <= kmax; kk++) {
        complex ss = abs_lambda * exp(complex<double>(0, (theta + 2*M_PI*kk)/alpha));
        s_star_struct s = {ss, (real(ss) + abs(ss))/2};
        temp_s_star_vec[kk - kmin] = s;
        //cout << "temp_s_star_vec[kk - kmin].s_star " << temp_s_star_vec[kk - kmin].s_star << endl;
        //cout << "temp_s_star_vec[kk - kmin].phi_star " << temp_s_star_vec[kk - kmin].phi_star << endl;
    }

    //parabolic contour must satisfy this condition
    sort(temp_s_star_vec.begin(), temp_s_star_vec.end(), [](s_star_struct a, s_star_struct b)
        {return a.phi_star < b.phi_star;});
    

    for (int i = 0; i < temp_s_star_vec.size(); i++) {
        if(temp_s_star_vec[i].phi_star > 1e-15) {
            s_star_vec [i] = temp_s_star_vec[i];
        }
    }
    
    s_star_vec.insert(s_star_vec.begin(), {complex<double>(0,0), 0});
    s_star_vec.erase( unique( s_star_vec.begin(), s_star_vec.end() ), s_star_vec.end() );

    for(int i = 0; i < s_star_vec.size(); i++) {
        //cout << "s_star_vec[i].s_star " << s_star_vec[i].s_star << endl;
        //cout << "s_star_vec[i].phi_star " << s_star_vec[i].phi_star << endl;
    }

    int J1 = s_star_vec.size();
    int J = J1 - 1;
    vector<double> p = vector<double>(J+1, k + 1);
    if(abs(lambda) <= 1 && abs(theta) > M_PI*0.9) 
        p[0] = max(0.0, -2*(-alpha*(k + 1)/2+alpha-beta+1));
    else
        p[0] = max(0.0, -2*(alpha - beta + 1));

    
    vector <double> q = vector<double>(J+1, k + 1);
    q[J] = std::numeric_limits<double>::infinity();
    s_star_vec.push_back(s_star_struct{complex<double>(std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), 
                                                        std::numeric_limits<double>::infinity()});

    vector<int> admissable_regions;
    for(int i = 0; i < s_star_vec.size() - 1; i++) {
        if(s_star_vec[i].phi_star < (log_epsilon - log_eps) && 
        s_star_vec[i].phi_star < s_star_vec[i+1].phi_star){
            admissable_regions.push_back(i);
        }
        
    }
    double JJ1 = admissable_regions.back();
    for (int i = 0; i < admissable_regions.size(); i++) {
        //cout << "admissable_regions i: " <<  i << " = " << admissable_regions[i] << endl;
    }
    vector <double> mu_vector = vector<double>(JJ1 + 1, std::numeric_limits<double>::infinity());
    vector <double> N_vector = vector<double>(JJ1 + 1, std::numeric_limits<double>::infinity());
    vector <double> h_vector = vector<double>(JJ1 + 1, std::numeric_limits<double>::infinity());
    
    bool find_region = false;
    vector<double>::iterator minElement;


    while(!find_region) {
        for(double j1 : admissable_regions) {
            double muJ1, NJ1, hJ1;
            if(j1 + 1 < J1) 
                optimal_param_RB(s_star_vec[j1].phi_star,
                                    s_star_vec[j1+1].phi_star, p[j1], q[j1], log_epsilon, &muJ1, &NJ1, &hJ1);
            
            else
                optimal_param_RU(s_star_vec[j1].phi_star, p[j1], log_epsilon, &muJ1, &NJ1, &hJ1);
            //cout << "muJ1 " << muJ1 << endl;
            //cout << "NJ1 " << NJ1 << endl;
            //cout << "hJ1 " << hJ1 << endl;
            mu_vector[j1] = muJ1;
            N_vector[j1] = NJ1;
            h_vector[j1] = hJ1;
        }
        minElement = std::min_element(std::begin(N_vector), std::end(N_vector));
        if(*minElement > 200){
            log_epsilon = log_epsilon +log(10);
            }
        else
            find_region = true;
    }

    int index = std::distance(std::begin(N_vector), minElement);

    //valor de u para o contorno
    double mu = mu_vector[index];
    //a grelha (?)
    double N = N_vector[index];
    //o passo
    double h = h_vector[index];

    //cout << "mu " << mu << endl;
    //cout << "N " << N << endl;
    //cout << "h " << h << endl;

    vector<complex<double>> u(abs(N*2));
    iota(u.begin(), u.end(), -abs(N));
    transform(u.begin(), u.end(), u.begin(),
               bind(multiplies<complex<double>>(), std::placeholders::_1, h * 1.0));

    //countour
    valarray<complex<double>> z = valarray<complex<double>>(u.size());
    //derivada de z
    vector<complex<double>> zd = vector<complex<double>>(u.size());
    //exp de z
    vector<complex<double>> expz = vector<complex<double>>(u.size());
    for(int i = 0; i < u.size(); i ++) {
        z[i] = pow(mu*(complex<double>(0,1) * u[i] + 1.0), 2);
        zd[i] = -2*mu*u[i] + 2*mu*complex<double>(0,1);
        expz[i] = exp(z[i]);
    }
    valarray<complex<double>> z1 = pow(z, alpha - beta);
    valarray<complex<double>> z2 = pow(z, alpha) - lambda;

    //a funcao H no paper
    vector<complex<double>> H = vector<complex<double>>(z.size());
    valarray<complex<double>> S = valarray<complex<double>>(z.size());
    for(int i = 0; i < z.size(); i++) {
        H[i] = z1[i]/z2[i] * zd[i];
        S[i] = expz[i] * H[i];
    }

    //cout << "S.sum()" << S.sum() << endl;

    complex<double> integral = h * S.sum() / 2.0 / M_PI / complex<double>(0,1);
    vector<s_star_struct> ss_star;

    for(int i = index + 1; i < s_star_vec.size() - 1; i++) {
        ss_star.push_back(s_star_vec[i]);
    }
    complex<double> res = residues(alpha, beta, ss_star);
    //cout << "integral " << integral << endl;
    //cout << "res " << res << endl;
    return integral + res;
    
}


dense_matrix calculate_MLF(dense_matrix A, double alpha, double beta) {

    //schur decomposition
    MatrixXd eigenMtxA = convertDenseEigenMtx(A);
    Eigen::MatrixXcd eigenMtxResult = Eigen::MatrixXcd::Constant(A.getRowVal(), A.getColVal(), 0);
    ComplexSchur<MatrixXd> schur(eigenMtxA);
    MatrixXcd T = schur.matrixT();
    MatrixXcd U = schur.matrixU();
    

    dense_matrix E = dense_matrix(A.getRowVal(), A.getColVal());

    //target accuracy threshold
    double tau = 1.0e-14;
    
    //maximal argument for the gamma function
    double max_gamma_arg = 171.624;

    //upper bound for the number of terms in the series expansion
    double Jmax = floor((max_gamma_arg - beta)/alpha);

    double numerator = tgamma(alpha*Jmax + beta);
    double denominator = 1; //falling factorial of Jmax with k = 0
    double bound = pow((tau * numerator/denominator), 1/Jmax);

    //evaluate diagonal entries
    for(int col = 0; col < A.getColVal(); col++){
        bool accept = false;
        cout << "T(" << col << "," << col << "): " << T(col,col) << endl;
        if(abs(T(col,col)) < bound){
            eigenMtxResult(col, col) = series_expansion(T(col,col), alpha, beta, Jmax, &accept);
            cout << "series expansion for T(" << col << "," << col << "): " << eigenMtxResult(col,col) << endl;
            cout << "accept: " << accept << endl;
        }


        if(!accept){
            if(abs(T(col,col)) <= EPS){
                //cout <<"less than eps" << endl;
                eigenMtxResult(col, col) = 1/tgamma(beta);
                cout << "small val, eigenMtxResult(" << col << "," << col << "): " << eigenMtxResult(col,col) << endl;
            }
            else {
                //cout << "doing lti" << endl;
                complex<double> LTI_value = LTI(T(col,col), alpha, beta);
                eigenMtxResult(col, col) = LTI_value;
                cout << "evaluated lti for T(" << col << "," << col << "): " << eigenMtxResult(col,col) << endl;
            }
        }
    }

    //parlett recurrence
    for(int col = 1; col < A.getColVal(); col++) {
        for(int row = 0; row < A.getRowVal()-1; row++){
            int i = row;
            int j = row + col;
            complex<double> result = 0;
            if(i < j && j < A.getColVal()){
                result += T(i,j) *((eigenMtxResult(i,i) - eigenMtxResult(j,j))/(T(i,i) - T(j,j)));
                for(int k = i + 1; k < j; k++){
                    result += (eigenMtxResult(i,k) * T(k,j) - T(i,k) * eigenMtxResult(k,j))/(T(i,i) - T(j,j));
                }
                eigenMtxResult(i,j) = result;
            }
        }

    }
    
    MatrixXcd realRes = U * eigenMtxResult * U.adjoint();

    E = convertEigenDenseMtx(realRes.real());
    

    return E;
}

