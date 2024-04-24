#include <cmath>
#include <vector>
#include <numeric>
#include <limits>
#include <complex>
#include <functional>
#include <bits/stdc++.h>
#include <omp.h>

#include "headers/utils.hpp"
#include "headers/Evaluate-Single-ML.hpp"
#include "mkl.h"

/*
Algorithm based on the paper "Computing the matrix Mittag–Leffler function with applications to fractional calculus"
by Roberto Garrappa and Marina Popolizio
 */


int maxJArgs;

//maximal argument for the gamma function
#define max_gamma_arg 171.624
#define tau 1.0e-14

using namespace std;


struct s_star_struct {
    complex<double> s_star;
    double phi_star = 0;

    bool operator==(const s_star_struct &other) const {
        return s_star == other.s_star && phi_star == other.phi_star;
    }
};


complex<double> residues(double alpha, double beta, vector<s_star_struct> s_star_vec, int k) {
    if(s_star_vec.empty())
        return {0,0};

    vector<double> omega = vector<double>(k+2);
    double pr = 1;

    for(int j = 1; j <= k + 1; j++) {
        double p = alpha - j + 1;
        pr *= p;
        omega[j] = pr/factorial(j);
    }

    //omega 1 -> alpha/1!
    //omega 2 -> alpha*(alpha-1)/2!
    //omega 3 -> alpha*(alpha-1)*(alpha-2)/3!

    vector<double> H_k = vector<double>(k+1);
    H_k[0] = 1;
    for(int j = 1; j <= k; j++) {
        double sum = 0;
        for(int l = 1; l <= j; l++) {
            sum += omega[l+1] * ( (double) (k*l)/j + 1) * H_k[j-l];
        }
        H_k[j] = (-1/alpha) * sum;
    }

    vector<double> coeff = vector<double>(k+1);

    for(int j = 0; j <= k; j++) {
        double sum = 0;
        for(int l = 0; l <= k - j; l++) {
            double numerator = falling_factorial(alpha-beta, l);
            sum += (numerator/factorial(l)) * H_k[k-j-l];
        }
        coeff[j] = (sum / factorial(j));
    }

    valarray<complex<double>> res = valarray<complex<double>>(s_star_vec.size());
    for(int i = 0; i < res.size(); i++) {
        //calculate polynomial P_k(s_star)
        complex<double> polyVal = complex<double>(0,0);
        for(int deg = 0; deg <= k; deg ++)
            polyVal += coeff[deg] * pow(s_star_vec[i].s_star, deg);

        complex<double> r1 =  1/pow(alpha, k+1) * exp( s_star_vec[i].s_star);
        complex<double> r2 = pow(s_star_vec[i].s_star, 1 - alpha*k -beta);
        res[i] = r1 * r2 * polyVal;
    }
    return res.sum();
}


void optimal_param_RB(double phi, double phi1, double p, double q, double log_epsilon, double * mu, double * N, double * h) {
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
    int admRegion;

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
            f_min = max(f_min,  1.5);
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
        *h = -2*PI/log_epsilon*(sqrt_phibar1-sqrt_phibar) /((1+w)*sqrt_phibar + sqrt_phibar1) ;
        *N = ceil(sqrt(1-log_epsilon/ *mu)/ *h) ;

    }
    else {
        *mu = 0;
        *N =  numeric_limits<double>::infinity();
        *h = 0;
    }


}


void optimal_param_RU(double phi, double p, double log_epsilon, double * mu, double * N, double * h) {
    double sqrt_phi = sqrt(phi);
    double phi_bar;
    double sqrt_phiBar;
    double sq_mu;
    double A;

    if(phi > 0)
        phi_bar =phi*1.01;
    else
        phi_bar = 0.01;

    sqrt_phiBar = sqrt(phi_bar);

    int f_min = 1 ; int f_max = 10 ; int f_tar = 5 ;

    bool stop = false;

    while(!stop) {
        double log_eps_phi = log_epsilon/phi_bar;
        *N = ceil(phi_bar/PI*(1 - 3*log_eps_phi/2 + sqrt(1-2*log_eps_phi)));
        A = PI * *N / phi_bar;
        sq_mu = sqrt_phiBar * abs(4 - A) / abs(7 - sqrt(1 + 12 * A));
        double fBar = pow(((sqrt_phiBar - sqrt_phi) / sq_mu), -p) ;
        stop = (p < 1.0e-14) || (f_min < fBar && fBar < f_max);
        if(!stop) {
            sqrt_phiBar = pow(f_tar, -1 / p) * sq_mu + sqrt_phi ;
            phi_bar = pow(sqrt_phiBar, 2) ;
        }
    }


    *mu = pow(sq_mu,2) ;
    *h = (-3*A - 2 + 2*sqrt(1+12*A)) / (4-A) / *N;

    double log_eps = log(EPS16);
    double threshold = log_epsilon - log_eps;
    double Q;

    if(*mu > threshold) {
        if(abs(p) < tau)
            Q = 0;
        else
            Q = pow(f_tar, -1/p) * sqrt(*mu);

        phi_bar = pow(Q + sqrt(phi),2);

        if(phi_bar < threshold) {
            double w = sqrt(log_eps/(log_eps-log_epsilon)) ;
            double u = sqrt(-phi_bar/log_eps) ;
            *mu = threshold;
            *N = ceil(w*log_epsilon/2/PI/(u*w-1));
            *h = sqrt(log_eps/(log_eps - log_epsilon))/ *N ;
        }
        else{
            *N =  numeric_limits<double>::infinity();
            *h = 0;
        }
    }

}

complex<double> calculateLTI(complex<double> lambda, double alpha, double beta, int k){

    if(abs(lambda) <= tau)
        return factorial(k)/tgamma(beta);

    double log_epsilon = log(1e-15);
    double log_eps = log(EPS16);

    //calculate phase angle of lambda
    double theta = arg(lambda);

    int kmin = ceil(-alpha/2 - theta/2/PI);
    int kmax = floor(alpha/2 - theta/2/PI);

    vector <s_star_struct> temp_s_star_vec = vector<s_star_struct>(kmax - kmin + 1);
    vector <s_star_struct> s_star_vec = vector<s_star_struct>(kmax - kmin + 1);

    //calculate values of s_star for valid values of j (kk)
    double abs_lambda = pow(abs(lambda), 1/alpha);
    for(int kk = kmin; kk <= kmax; kk++) {
        complex ss = abs_lambda * exp(complex<double>(0, (theta + 2*PI*kk)/alpha));
        s_star_struct s = {ss, (real(ss) + abs(ss))/2};
        temp_s_star_vec[kk - kmin] = s;
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

    int J1 = (int) s_star_vec.size();
    int J = J1 - 1;
    vector<double> p = vector<double>(J+1, k + 1);
    if(abs(lambda) <= 1 && abs(theta) > PI*0.9)
        p[0] = max( 0.0, -2*(-alpha*(k + 1)/2+alpha-beta+1));
    else
        p[0] = max(0.0, -2*(alpha - beta + 1));


    vector <double> q = vector<double>(J+1, k + 1);
    q[J] =  numeric_limits<double>::infinity();
    s_star_vec.push_back(s_star_struct{complex<double>( numeric_limits<double>::infinity(), numeric_limits<double>::infinity()),
                                        numeric_limits<double>::infinity()});

    vector<int> admissable_regions;
    for(int i = 0; i < s_star_vec.size() - 1; i++) {
        if(s_star_vec[i].phi_star < (log_epsilon - log_eps) && s_star_vec[i].phi_star < s_star_vec[i+1].phi_star){
        admissable_regions.push_back(i);
        }
    }
    double JJ1 = admissable_regions.back();
    vector <double> mu_vector = vector<double>((int) JJ1 + 1,  numeric_limits<double>::infinity());
    vector <double> N_vector = vector<double>((int) JJ1 + 1,  numeric_limits<double>::infinity());
    vector <double> h_vector = vector<double>((int) JJ1 + 1,  numeric_limits<double>::infinity());

    bool find_region = false;
    vector<double>::iterator minElement;


    while(!find_region) {
        for(int j1 : admissable_regions) {
            double muJ1, hJ1, NJ1;
            if(j1 + 1 < J1) {
                optimal_param_RB(s_star_vec[j1].phi_star,
                                 s_star_vec[j1 + 1].phi_star, p[j1], q[j1], log_epsilon, &muJ1, &NJ1, &hJ1);
            }

            else {
                optimal_param_RU(s_star_vec[j1].phi_star, p[j1], log_epsilon, &muJ1, &NJ1, &hJ1);
            }
            mu_vector[j1] = muJ1;
            N_vector[j1] = NJ1;
            h_vector[j1] = hJ1;
        }
        minElement =  min_element( begin(N_vector),  end(N_vector));
        if(*minElement > 200){
            log_epsilon = log_epsilon +log(10);
        }
        else
            find_region = true;
    }

    int index = (int) distance(begin(N_vector), minElement);


    double mu = mu_vector[index];
    int N = (int) N_vector[index];
    double h = h_vector[index];

    vector<double> u(abs(N*2) + 1);
    for(int i = -N; i <= N; i++) {
        u[i + N] = i * h;
    }

    //contour
    complex<double> z = 0;
    complex<double> zd = 0;
    complex<double> expZ = 0;
    complex<double> z1 = 0;
    complex<double> z2 = 0;
    complex<double> z3 = 0;
    complex<double> z4 = 0;
    complex<double> F = 0;
    complex<double> sum = 0;

    for(auto uVal : u) {
        //z = mu*(1i*u+1).^2 ; -> z = mu * (1 + uVal i)^2 = mu * (1 + uVal i) * (1 + uVal i)
        // z.real = mu * (1 - uVal * uVal) ; z.imag = mu * (uVal + uVal)
        z = {mu * (1 - uVal * uVal), mu * 2 * uVal};
        zd = {-2 * mu * uVal,  2 * mu};
        expZ = exp(z);
        z1 = pow(z, alpha - beta);
        z2 = pow(z, alpha);
        z3 = z2 - lambda;
        z4 = pow(z3, k + 1);
        F = z1 / z4;
        F *= zd;
        sum += expZ * F;
    }

    complex<double> integral = h * sum / 2.0 / PI / complex<double>(0,1);

    vector<s_star_struct> ss_star;

    for(int i = index + 1; i < s_star_vec.size() - 1; i++) {
        ss_star.push_back(s_star_vec[i]);
    }
    complex<double> res = residues(alpha, beta, ss_star, k);
    return factorial(k) * (integral + res);

}


complex<double> LTI(complex<double> lambda, double alpha, double beta, int k) {
    complex<double> result = 0;

    int p;

    if(k <=3)
        p = 0;
    else if (k <= 7)
        p = 1;
    else
        p = 2;

    //calculate coefficients
    vector<vector<double>> c(k - p + 1, vector<double>(k - p + 1, 0.0));


    c[0][0] = 1;
    for (int kk = 1; kk <= k - p; kk++) {
        c[kk][0] = (1 - alpha * (kk - 1) - beta) * c[kk - 1][0];
        for (int j = 1; j < kk; j++) {
            c[kk][j] = c[kk - 1][j - 1] + (j + 1 - alpha * (kk - 1) - beta) * c[kk - 1][j];
        }
        c[kk][kk] = 1;
    }

    //#pragma omp parallel for reduction(+:result) schedule(guided) if (k > 1)
    for (int j = 0; j <= k - p; j++) {
        if (abs(c[k - p][j]) > tau) {
            result += c[k - p][j] * calculateLTI(lambda, alpha, (k - p) * alpha + beta - j, p);
        }
    }

    result = result / pow(alpha, k - p);

    return result;
}

complex<double> series_expansion(complex<double> z, double alpha, double beta, bool * accept, int kd) {

    vector<complex<double>> sumArgs = vector<complex<double>>(maxJArgs + 1 - kd);
    vector<double> absSumArgs = vector<double>(maxJArgs + 1 - kd);
    vector<bool> iAbsSumArg = vector<bool>();
    complex<double> result = complex<double>(0,0);

    double numerator = 1.0;

    if (abs(z) < EPS16){
        *accept = true;
        return factorial(kd)/tgamma(alpha*kd + beta);
    }

    for(int j = kd; j <= maxJArgs; j++){
        double denominator = tgamma( (alpha*j + beta));
        if(kd > 0)
            numerator = falling_factorial(j, kd);
        complex<double> sumVal = (numerator/denominator);
        sumVal *= pow(z, j - kd);
        result += sumVal;
        sumArgs[j - kd] = sumVal;
        absSumArgs[j - kd] =  abs(sumVal);
    }


    int count = 0;
    for (const auto& val : absSumArgs) {
        if(val > EPS16 / 2) {
            iAbsSumArg.push_back(true);
            count++;
        }
    }

    if (!count) {
        iAbsSumArg.push_back(true);
    }


    vector<double> filterAbsSumArg;
    vector<complex<double>> filterSumArg;
    for (int i = 0; i < absSumArgs.size(); ++i) {
        if (iAbsSumArg[i]) {
            filterAbsSumArg.push_back(absSumArgs[i]);
            filterSumArg.push_back(sumArgs[i]);
        }
    }


    vector<size_t> indices(filterAbsSumArg.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return filterAbsSumArg[i] < filterAbsSumArg[j];
    });

    vector<double> sortAbsSumArg(filterAbsSumArg.size());
    vector<complex<double>> sortSumArg(filterSumArg.size());
    for (size_t i = 0; i < filterAbsSumArg.size(); ++i) {
        sortAbsSumArg[i] = filterAbsSumArg[indices[i]];
        sortSumArg[i] = filterSumArg[indices[i]];
    }

     vector<complex<double>> S(sortSumArg.size());

    if (sortSumArg.size() != 1) {
         partial_sum(sortSumArg.begin(), sortSumArg.end(), S.begin());
        S.erase(S.begin());
    }

    // Calculate Err_Round
    double Err_Round1, Err_Round2, Err_Round;
    int J = (int) sortSumArg.size() - 1;
     vector<int> JJ;
    JJ.push_back(J);
    for (int i = J; i >= 1; --i) {
        JJ.push_back(i);
    }

    //Err_Round1
    double sumJJAbs = 0.0;
    for (size_t i = 0; i < sortAbsSumArg.size(); ++i) {
        sumJJAbs += JJ[i] * abs(sortAbsSumArg[i]);
    }
    Err_Round1 = sumJJAbs * EPS16;

    //Err_Round2
    if (filterSumArg.size() == 1) {
        Err_Round2 = Err_Round1;
    } else {
        Err_Round2 = accumulate(S.begin(), S.end(), 0.0, [](double acc, const complex<double>& val) {
            return acc + abs(val);});
        Err_Round2 *= EPS16;
    }


    Err_Round = exp((log(Err_Round1) + log(Err_Round2)) / 2);

    //i_z_se_accept = (Err_Round./(e+abs(E_se)) < tau) & ~(E_se==0)  ;
    *accept = (Err_Round / (1 + abs(result)) < tau);

    return result;
}

complex<double> evaluateSingle(complex<double> z, double alpha, double beta, int k) {

    complex<double> result;
    bool accept = false;

    maxJArgs = floor((max_gamma_arg - beta)/alpha);

    double numerator = tau * tgamma(alpha*maxJArgs + beta);
    double denominator = falling_factorial(maxJArgs, k);
    double exp = 1.0/(maxJArgs - k);
    double bound = pow((numerator/denominator), exp);

    complex<double> tVal = z;

    if(abs(tVal) <= bound){
        result = series_expansion(tVal, alpha, beta, &accept, k);
    }

    if(!accept){
        result = LTI(tVal, alpha, beta, k);
    }

    if(tVal.imag() == 0){
        result = {result.real(), 0};
    }

    return result;
}