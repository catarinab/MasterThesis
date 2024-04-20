mpi = mpiicpx
C++ = g++
CFLAGS= -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -O2
CFLAGSDISTR= -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
CFLAGSEigen= -I ../eigen
CFLAGSAutoDiff= -I ../autodiff
DEBUG = -g
TARGET=MLF-distr
TARGETSHARED=MLF-shared
TARGETTEST=test.out
TARGETFRACSHARED=Frac-shared.out
TARGETTESTEigen=testEigen.out
TARGETHESS=hess.out
LINK=  -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[0;33m
NORMAL=\033[0m


all: test hess

test:
	@echo "$(YELLOW)Making testing version: $(TARGETTEST) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) -I ../eigen -c ../utils/schur-blocking.cpp -o schur-blocking.o
	$(C++) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) -c ../utils/calculate-MLF.cpp -o calculate-MLF.o
	$(C++) $(CFLAGS) -c ../utils/Evaluate-Single-ML.cpp -o Evaluate-Single-ML.o
	$(C++) $(CFLAGS) -c test.cpp -o test.o
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o mtx_ops_mkl.o calculate-MLF.o Evaluate-Single-ML.o test.o -o $(TARGETTEST)
	rm *.o

frac:
	@echo "$(GREEN)Making fractional Arnoldi version: $(TARGETFRACSHARED) $(NORMAL)"
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) -c ../utils/mtx_ops_mkl.cpp -o mtx_ops_mkl.o
	$(C++) $(CFLAGS) -c ../utils/arnoldiIteration-shared.cpp -o arnoldiIteration-shared.o
	$(C++) $(CFLAGS) $(CFLAGSAutoDiff) $(CFLAGSEigen) -c ../utils/Frac-Utils.cpp -o Frac-Utils.o
	$(C++) $(CFLAGS) $(CFLAGSEigen) $(CFLAGSAutoDiff) -c FractionalKrylov.cpp -o FractionalKrylov.o
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o mtx_ops_mkl.o arnoldiIteration-shared.o Frac-Utils.o FractionalKrylov.o -o $(TARGETFRACSHARED)
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
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o schur-blocking.o mtx_ops_mkl.o Evaluate-Single-ML.o calculate-MLF.o test.o -o $(TARGETTESTEigen)
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
	$(C++) $(CFLAGS) $(LINK) csr_matrix.o dense_vector.o io_ops.o dense_matrix.o mtx_ops_mkl.o arnoldiIteration-shared.o generateHessenberg.o mmio.o -o $(TARGETHESS)
	rm *.o


clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -f *.o $(TARGET) $(TARGETSHARED) $(TARGETTEST) $(TARGETTESTEigen) $(TARGETHESS) $(TARGETFRACSHARED)


