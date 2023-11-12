#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

#ifndef MTX_OPS
    #define MTX_OPS 1
    #include "mtx_ops.cpp"
#endif

using namespace std;

int * displs;
int * counts;
int helpSize = 0;

void initGatherVars(int size, int nprocs) {
    helpSize = size/nprocs;
    displs = (int *)malloc(nprocs*sizeof(int)); 
    counts = (int *)malloc(nprocs*sizeof(int)); 

    if(nprocs == 1){
        displs[0] = 0;
        counts[0] = size;
        return;
    }

    for(int i = 0; i < nprocs; i++) {
        displs[i] = i* helpSize; 
        counts[i] = helpSize;
    }
    counts[nprocs - 1] += size % nprocs;

}

void sendVectors(DenseVector a, DenseVector b, int helpSize, int func, int me, int size, int nprocs) {

    MPI_Bcast(&helpSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if(func == MV)
        MPI_Bcast(&a.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    else {
        MPI_Scatterv(&a.values[0], counts, displs, MPI_DOUBLE, &a.values[0], helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Scatterv(&b.values[0], counts, displs, MPI_DOUBLE, &b.values[0], helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
}

double distrDotProduct(DenseVector a, DenseVector b, int size, int me, int nprocs) {
    double dotProd = 0;

    sendVectors(a, b, helpSize, VV, me, size, nprocs);
    
    double temp = dotProduct(a, b, 0, helpSize);

    MPI_Reduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    return dotProd;
    
}

DenseVector distrSubOp(DenseVector a, DenseVector b, int size, int me, int nprocs) {
    DenseVector finalRes(size); 

    sendVectors(a, b, helpSize, SUB, me, size, nprocs);

    DenseVector res = subtractVec(a, b, 0, helpSize);

    MPI_Gatherv(&res.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    return finalRes;
}

DenseVector distrSumOp(DenseVector a, DenseVector b, int size, int me, int nprocs) {
    DenseVector finalRes(size); 

    sendVectors(a, b, helpSize, ADD, me, size, nprocs);

    DenseVector res = addVec(a, b, 0, helpSize);

    MPI_Gatherv(&res.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    return finalRes;
}

DenseVector distrMatrixVec(CSR_Matrix A, DenseVector vec, int size, int me, int nprocs) {
    
    DenseVector finalRes(size);

    sendVectors(vec, DenseVector(0), helpSize, MV, me, size, nprocs);

    DenseVector res = sparseMatrixVector(A, vec, 0, helpSize, size);

    MPI_Gatherv(&res.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    return finalRes;
}