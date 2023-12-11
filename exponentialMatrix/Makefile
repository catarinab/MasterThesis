C++ = mpic++
CFLAGS=-fopenmp -std=c++17 -I ../eigen 
TARGET=exp

all:
	$(C++) $(CFLAGS) -c ../utils/csr_matrix.cpp -o csr_matrix.o
	$(C++) $(CFLAGS) -c ../utils/dense_vector.cpp -o dense_vector.o
	$(C++) $(CFLAGS) -c ../utils/dense_matrix.cpp -o dense_matrix.o
	$(C++) $(CFLAGS) -c ../utils/distr_mtx_ops.cpp -o distr_mtx_ops.o
	$(C++) $(CFLAGS) -c ../utils/help_proccess.cpp -o help_proccess.o
	$(C++) $(CFLAGS) -c ../utils/io_ops.cpp -o io_ops.o
	$(C++) $(CFLAGS) -c ../utils/pade_approx_utils.cpp -o pade_approx_utils.o
	$(C++) $(CFLAGS) -c ../utils/mtx_ops.cpp -o mtx_ops.o
	$(C++) $(CFLAGS) -c exponentialMtx.cpp -o exp-mtx.o
	$(C++) $(CFLAGS) exp-mtx.o csr_matrix.o dense_vector.o dense_matrix.o distr_mtx_ops.o help_proccess.o io_ops.o pade_approx_utils.o mtx_ops.o -o $(TARGET)
	rm *.o

clean:
	rm $(TARGET)

