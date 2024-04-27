#ifndef FRAC_UTILS_HPP
#define FRAC_UTILS_HPP

#include <random>
#include <chrono>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/real/real.hpp>

#include "dense_vector.hpp"
#include "dense_matrix.hpp"

using Eigen::VectorXd;


autodiff::real totalRNG(const autodiff::ArrayXreal& q, const autodiff::ArrayXreal& u, autodiff::real t);

autodiff::real spectralKernelRNG(autodiff::real alpha, autodiff::real gamma, autodiff::real u);

autodiff::real stblrndsub(autodiff::real alpha, autodiff::real u1, autodiff::real u2);

autodiff::real stblrnd(autodiff::real alpha, autodiff::real beta, autodiff::real gamma, autodiff::real delta,
                       autodiff::real u1, autodiff::real u2);

dense_vector dTotalRNG(double alpha, double gamma, autodiff::real u1, autodiff::real u2, autodiff::real u3,
                       autodiff::real t);



#endif // FRAC_UTILS_HPP

