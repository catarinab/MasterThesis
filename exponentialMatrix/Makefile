mpi = mpiicpx
C++ = g++
CFLAGS= -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
CFLAGSEigen= -I ../eigen
TARGET=exp
TARGETSHARED=exp-shared
LINK=  -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
GREEN=\033[0;32m
RED=\033[0;31m
NORMAL=\033[0m


all: distr shared

distr:
	@echo "$(GREEN)Making distributed memory version: $(TARGET) $(NORMAL)"
	$(mpi) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(mpi) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(mpi) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(mpi) $(CFLAGS) -c ../utils/arnoldiIteration.cpp -o arnoldiIteration.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c ../utils/distr_mtx_ops.cpp -o distr_mtx_ops.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c ../utils/help_process.cpp -o help_process.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c ../utils/io_ops.cpp -o io_ops.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c ../utils/pade_exp_approx.cpp -o pade_exp_approx.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(mpi) $(CFLAGS) $(CFLAGSEigen) -c exponentialMtx.cpp -o exp-mtx.o
	$(mpi) $(CFLAGS) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldiIteration.o distr_mtx_ops.o help_process.o io_ops.o pade_exp_approx.o mtx_ops_mkl.o -o $(TARGET)
	rm *.o

shared:
	@echo "$(GREEN)Making shared memory version: $(TARGETSHARED) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/pade_exp_approx.cpp -o pade_exp_approx.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) -c exponentialMtx-shared.cpp -o exp-mtx.o
	$(C++) $(CFLAGS) $(LINK) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o arnoldiIteration-shared.o io_ops.o pade_exp_approx.o mtx_ops_mkl.o -o $(TARGETSHARED)
	rm *.o

clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm *.o $(TARGET) $(TARGETSHARED)


