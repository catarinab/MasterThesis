mpi = mpiicpx
C++ = g++
CFLAGS= -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers -O3
CFLAGSDISTR= -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers -O3
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
	$(mpi) $(CFLAGSDISTR) -c ../utils/arnoldi_iteration.cpp -o arnoldi_iteration.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/distr_mtx_ops.cpp -o distr_mtx_ops.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/help_process.cpp -o help_process.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/io_ops.cpp -o io_ops.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/scaling_and_squaring.cpp -o scaling_and_squaring.o
	$(mpi) $(CFLAGSDISTR) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(mpi) $(CFLAGSDISTR) -c exponential_mtx.cpp -o exp-mtx.o
	$(mpi) $(CFLAGSDISTR) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldi_iteration.o \
		distr_mtx_ops.o help_process.o io_ops.o scaling_and_squaring.o mtx_ops_mkl.o -o $(TARGET)
	rm *.o

shared:
	@echo "$(GREEN)Making shared memory version: $(TARGETSHARED) $(NORMAL)"
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/arnoldi_iteration_shared.cpp -o arnoldi_iteration_shared.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/scaling_and_squaring.cpp -o scaling_and_squaring.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(DEBUG) $(CFLAGS) -c exponential_mtx_shared.cpp -o exp-mtx.o
	$(C++) $(DEBUG) $(CFLAGS) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldi_iteration_shared.o \
		io_ops.o scaling_and_squaring.o mtx_ops_mkl.o -o $(TARGETSHARED)
	rm *.o

clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -f *.o $(TARGET) $(TARGETSHARED) $(TARGETTEST)


