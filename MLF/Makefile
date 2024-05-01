mpi = mpiicpx
C++ = g++
CFLAGS= -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -march=native -O3
CFLAGSDISTR= -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -march=native -O3
CFLAGSEigen= -I ../eigen
CFLAGSAutoDiff= -I ../autodiff
CFLAGSCubature= -I ../cubature
DEBUG = -g
TARGET=MLF-distr
TARGETSHARED=mlf-shared.out
TARGETTEST=test.out
TARGETFRACSHARED=frac.out
TARGETTESTEigen=testEigen.out
TARGETHESS=hess.out
LINK=  -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lgomp -lpthread -lm -ldl -ltbb
GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[0;33m
NORMAL=\033[0m


all: test hess

test:
	@echo "$(YELLOW)Making testing version: $(TARGETTEST) $(NORMAL)"
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(DEBUG) $(CFLAGS) -I ../eigen -c ../utils/schur-blocking.cpp -o schur-blocking.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/calculate-MLF.cpp -o calculate-MLF.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/Evaluate-Single-ML.cpp -o Evaluate-Single-ML.o
	$(C++) $(DEBUG) $(CFLAGS) -c test.cpp -o test.o
	$(C++) $(CFLAGS) $(LINK) arnoldiIteration-shared.o csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o \
	mtx_ops_mkl.o calculate-MLF.o Evaluate-Single-ML.o test.o -o $(TARGETTEST)
	rm *.o

mlf-shared:
	@echo "$(YELLOW)Making testing version: $(TARGETTEST) $(NORMAL)"
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(DEBUG) $(CFLAGS) -I ../eigen -c ../utils/schur-blocking.cpp -o schur-blocking.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/calculate-MLF.cpp -o calculate-MLF.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/Evaluate-Single-ML.cpp -o Evaluate-Single-ML.o
	$(C++) $(DEBUG) $(CFLAGS) -c mlf-shared.cpp -o mlf-shared.o
	$(C++) $(DEBUG) $(CFLAGS) $(LINK) arnoldiIteration-shared.o csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o \
	mtx_ops_mkl.o calculate-MLF.o Evaluate-Single-ML.o mlf-shared.o -o $(TARGETSHARED)
	rm *.o

mlf-shared-parallel:
	@echo "$(YELLOW)Making testing version: $(TARGETTEST) $(NORMAL)"
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(DEBUG) $(CFLAGS) -I ../eigen -c ../utils/schur-blocking.cpp -o schur-blocking.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/calculate-MLF.cpp -o calculate-MLF.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/Evaluate-Single-ML-parallel.cpp -o Evaluate-Single-ML.o
	$(C++) $(DEBUG) $(CFLAGS) -c mlf-shared.cpp -o mlf-shared.o
	$(C++) $(DEBUG) $(CFLAGS) $(LINK) arnoldiIteration-shared.o csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o \
	mtx_ops_mkl.o calculate-MLF.o Evaluate-Single-ML.o mlf-shared.o -o mlf-shared-parallel.out
	rm *.o

frac:
	@echo "$(GREEN)Making fractional Arnoldi version: $(TARGETFRACSHARED) $(NORMAL)"
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) -c ../utils/pade_exp_approx.cpp -o pade_exp_approx.o
	$(C++) $(DEBUG) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(DEBUG) $(CFLAGS) $(CFLAGSAutoDiff) $(CFLAGSEigen) -c ../utils/fractionalKrylov.cpp -o fractionalKrylov.o
	$(C++) $(DEBUG) $(CFLAGS) $(CFLAGSCubature) $(CFLAGSAutoDiff) $(CFLAGSEigen) -c frac-krylov.cpp -o frac-krylov.o
	$(C++) $(DEBUG) $(CFLAGSCubature) $(CFLAGS) $(LINK) ../cubature/hcubature.o csr_matrix.o dense_vector.o io_ops.o dense_matrix.o \
	mtx_ops_mkl.o pade_exp_approx.o arnoldiIteration-shared.o fractionalKrylov.o frac-krylov.o -o $(TARGETFRACSHARED)
	rm *.o

testEigen:
	@echo "$(YELLOW)Making eigen version: $(TARGETTEST) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) -c ../utils/schur-blocking.cpp -o schur-blocking.o
	$(C++) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) -c ../utils/Evaluate-Single-ML.cpp -o Evaluate-Single-ML.o
	$(C++) $(CFLAGS) -I ../eigen -c ../utils/calculate-MLF-Eigen.cpp -o calculate-MLF.o
	$(C++) $(CFLAGS) -c test.cpp -o test.o
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o mtx_ops_mkl.o \
	Evaluate-Single-ML.o calculate-MLF.o test.o -o $(TARGETTESTEigen)
	rm *.o
hess:
	@echo "$(YELLOW)Making hessenberg generating program: $(TARGETHESS) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(CFLAGS) -c ../utils/mmio.cpp -o mmio.o
	$(C++) $(CFLAGS) -c generateHessenberg.cpp -o generateHessenberg.o
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o mtx_ops_mkl.o arnoldiIteration-shared.o \
	generateHessenberg.o mmio.o -o $(TARGETHESS)
	rm *.o


clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -f *.o $(TARGET) $(TARGETSHARED) $(TARGETTEST) $(TARGETTESTEigen) $(TARGETHESS) $(TARGETFRACSHARED)


