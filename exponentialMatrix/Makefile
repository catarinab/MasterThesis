mpi = mpiicpx
C++ = g++
CFLAGS= -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers
CFLAGSDISTR= -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers
DEBUG = -g
CFLAGSEigen= -I ../eigen
TARGET=exp
TARGETSHARED=exp-shared
TARGETTEST=test
LINK=  -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[0;33m
NORMAL=\033[0m


all: distr shared

distr:
	@echo "$(GREEN)Making distributed memory version: $(TARGET) $(NORMAL)"
	$(mpi) $(CFLAGSDISTR) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/arnoldiIteration.cpp -o arnoldiIteration.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c ../utils/distr_mtx_ops.cpp -o distr_mtx_ops.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c ../utils/help_process.cpp -o help_process.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c ../utils/io_ops.cpp -o io_ops.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c ../utils/pade_exp_approx.cpp -o pade_exp_approx.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(mpi) $(CFLAGSDISTR) $(CFLAGSEigen) -c exponentialMtx.cpp -o exp-mtx.o
	$(mpi) $(CFLAGSDISTR) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldiIteration.o distr_mtx_ops.o help_process.o io_ops.o pade_exp_approx.o mtx_ops_mkl.o -o $(TARGET)
	rm *.o

shared:
	@echo "$(GREEN)Making shared memory version: $(TARGETSHARED) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/arnoldi-MKL.cpp -o arnoldiIteration-shared.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/pade_exp_approx.cpp -o pade_exp_approx.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c exponentialMtx-shared.cpp -o exp-mtx.o
	$(C++) $(CFLAGS) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldiIteration-shared.o io_ops.o pade_exp_approx.o mtx_ops_mkl.o -o $(TARGETSHARED)
	rm *.o

clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -f *.o $(TARGET) $(TARGETSHARED) $(TARGETTEST)


