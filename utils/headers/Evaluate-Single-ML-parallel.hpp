#include <complex>

#include "mkl.h"

std::complex<double> evaluateSingle(std::complex<double> z, double alpha, double beta, int k, int numThreads = 0);