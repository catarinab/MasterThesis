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

    displs[0] = 0;
    counts[0] = nprocs == 1 ? size : 0;

    if(nprocs != 1)
        for(int i = 1; i < nprocs; i++) {
            displs[i] = (i - 1) * helpSize; 
            counts[i] = helpSize;
        }
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
    
    double temp = dotProduct(a, b, (nprocs - 1) * helpSize, size);

    MPI_Reduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    return dotProd;
    
}

DenseVector distrSubOp(DenseVector a, DenseVector b, int size, int me, int nprocs) {
    
    DenseVector res;
    DenseVector finalRes(size); 

    sendVectors(a, b, helpSize, SUB, me, size, nprocs);

    res = subtractVec(a, b, (nprocs - 1) * helpSize, size);

    MPI_Gatherv(&finalRes.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0)
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    

    return finalRes;
}

DenseVector distrSumOp(DenseVector a, DenseVector b, int size, int me, int nprocs) {
    
    DenseVector res;
    DenseVector finalRes(size); 

    sendVectors(a, b, helpSize, ADD, me, size, nprocs);

    res = addVec(a, b, (nprocs - 1) * helpSize, size);


    MPI_Gatherv(&finalRes.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0)
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    
    return finalRes;
}

DenseVector distrMatrixVec(CSR_Matrix A, DenseVector vec, int size, int me, int nprocs) {
    
    DenseVector finalRes(size);

    sendVectors(vec, DenseVector(0), helpSize, MV, me, size, nprocs);

    DenseVector res;
    res = sparseMatrixVector(A, vec, (nprocs - 1) * helpSize, size, size);

    MPI_Gatherv(&finalRes.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0){
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    }

    return finalRes;
}