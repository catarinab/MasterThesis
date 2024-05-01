#ifndef MLF_LTI_HPP
#define MLF_LTI_HPP

#include <complex>

std::complex<double> evaluateSingle(std::complex<double> tVal, double alpha, double beta, int k, int nrThreads = 0);

#endif