#ifndef HELP_PROCCESS_HPP
#define  HELP_PROCCESS_HPP

#include "csr_matrix.hpp"

void helpProcess(const csr_matrix& A, int me, int size, int func, int * displs, int * counts);

#endif // HELP_PROCCESS_HPP