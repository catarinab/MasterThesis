#include <random>
#include <chrono>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include "headers/fractionalKrylov.hpp"
#include "headers/utils.hpp"
#include "headers/dense_vector.hpp"

using namespace autodiff;
using Eigen::VectorXd;


autodiff::real spectralKernelRNG(autodiff::real alpha, autodiff::real gamma, autodiff::real u) {
    //abs(tan((α*pi)*u + atan(1/tan(pi*α)))*λ*sin(pi*α)-λ*cos(pi*α))^(1/α)
    autodiff::real val = abs(tan(alpha * M_PI * u + atan(1 / tan(M_PI * alpha))) *
            gamma * sin(M_PI * alpha) - gamma * cos(M_PI * alpha));
    return pow(val, 1 / alpha);
}

autodiff::real totalRNG(const autodiff::ArrayXreal& q, const autodiff::ArrayXreal& u, autodiff::real t) {
    auto alpha = q(0);
    auto gamma = q(1);
    auto u1 = u(0);
    auto u2 = u(1);
    auto u3 = u(2);
    autodiff::real nu = ceil((gamma/alpha).val());
    autodiff::real result =  pow(spectralKernelRNG(alpha, pow(t, alpha), u1),(alpha * nu) / gamma)
            * stblrndsub(gamma/(alpha*nu), u2, u3);
    return result;
}


dense_vector dTotalRNGp(double alpha, double gamma, autodiff::real u1, autodiff::real u2, autodiff::real u3, autodiff::real t) {
    autodiff::ArrayXreal p(2);
    p << alpha, gamma;
    autodiff::ArrayXreal u(3);
    u << u1, u2, u3;

    VectorXd gx = gradient(totalRNG, wrt(p), at(p, u, t));

    dense_vector result = dense_vector((int) gx.size());
    for (int i = 0; i < gx.size(); i++)
        result.insertValue(i, gx(i));

    return result;
}

// subordinator
autodiff::real stblrndsub(autodiff::real alpha, autodiff::real u1, autodiff::real u2) {
    //stblrnd(alpha,one(eltype(alpha)),cos(alpha*pi/2)^(1/alpha),zero(eltype(alpha)),U1,U2)
    autodiff::real cos_val = cos(alpha * M_PI / 2);
    autodiff::real pow_val = pow(cos_val, 1.0 / alpha);
    return autodiff::real(stblrnd(alpha, 1, pow_val, 0, u1, u2));
}

// adapted from https://github.com/markveillette/stbl/blob/master/stblrnd.m
autodiff::real stblrnd(autodiff::real alpha, autodiff::real beta, autodiff::real gamma, autodiff::real delta,
                          autodiff::real u1, autodiff::real u2) {
    autodiff::real r;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> normalDistribution(0.0, 1.0);

    if(alpha <= 0 || alpha > 2) {
        throw std::invalid_argument("Alpha must be a scalar which lies in the interval (0,2]");
    }
    if(abs(beta) > 1) {
        throw std::invalid_argument("Beta must be a scalar which lies in the interval [-1,1]");
    }
    if(gamma < 0) {
        throw std::invalid_argument("Gamma must be a non-negative scalar");
    }

    //Gaussian Distribution
    if(alpha == 2) {
        r = sqrt(2) * normalDistribution(generator);
    }

        //Cauchy Distribution
    else if(alpha == 1 && beta == 0) {
        r = tan(M_PI /2 * (2 * u1 - 1));
    }

        //Levy distribution (a.k.a. Pearson V)
    else if (alpha == 0.5 && abs(beta) == 1) {
        r = beta / pow(normalDistribution(generator), 2);
    }

        //Symmetric alpha-stable
    else if (beta == 0) {
        autodiff::real V = M_PI/2 * (2*u1 - 1);
        autodiff::real W = -log(u2);
        r = sin(alpha * V) / pow(cos(V), 1/alpha) * pow(cos(V * (1 - alpha)) / W, (1 - alpha) / alpha);
    }

    //General case, alpha not 1
    else if (alpha != 1) {
        autodiff::real V = M_PI/2 * (2*u1 - 1);
        autodiff::real W = -log(u2);
        autodiff::real C = beta * tan(M_PI * alpha / 2);
        autodiff::real B = atan(C);
        autodiff::real S = pow(1 + C * C, 1/(2*alpha));
        r = S * sin(alpha * V + B) / pow(cos(V), 1/alpha) * pow(cos((1 - alpha) * V - B) / W, (1 - alpha) / alpha);
    }

        //General case, alpha = 1
    else {
        autodiff::real V = M_PI/2 * (2*u1 - 1);
        autodiff::real W = -log(u2);
        autodiff::real piover2 = M_PI/2;
        autodiff::real sclshftV =  piover2 + beta * V ;
        r = 1/piover2 * ( sclshftV * tan(V) - beta * log( (piover2 * W * cos(V) ) / sclshftV ) );
    }

    //scale and shift
    if(alpha != 1) {
        return gamma * r + delta;
    }
    else {
        return gamma * r + (2/M_PI) * beta * gamma * log(gamma) + delta;
    }

}
