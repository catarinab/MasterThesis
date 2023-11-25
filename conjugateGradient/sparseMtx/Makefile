C++ = mpic++
CFLAGS=-fopenmp -std=c++17 -I ../../eigen 
TARGET=cg

all:
	$(C++) $(CFLAGS) conjugateGradient.cpp -o $(TARGET)

clean:
	rm $(TARGET)

