#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#ifndef MTX_OPS
    #define MTX_OPS 1
    #include "mtx_ops.cpp"
#endif

using namespace std;

int * displs;
int * counts;
int helpSize = 0;

void initGatherVars(int size, int nprocs) {
    displs = (int *)malloc(nprocs*sizeof(int)); 
    counts = (int *)malloc(nprocs*sizeof(int)); 
    helpSize = size/nprocs;

    displs[0] = 0;
    counts[0] = 0; 

    for(int i = 1; i < nprocs; i++) {
        displs[i] = (i - 1) * helpSize; 
        counts[i] = helpSize;
    }
}


void sendVectors(Vector a, Vector b, int helpSize, int func, int me, int size, int nprocs) {
    int temp = 0;

    MPI_Bcast(&helpSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if(func == MV)
        for(int proc = 1; proc < nprocs ; proc++){
            int end = displs[proc] + counts[proc];
            
            MPI_Send(&a.values[0], size, MPI_DOUBLE, proc, FUNCTAG, MPI_COMM_WORLD);
            MPI_Send(&displs[proc], 1, MPI_INT, proc, FUNCTAG, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, proc, FUNCTAG, MPI_COMM_WORLD);
        }

    else {
        MPI_Scatterv(&a.values[0], counts, displs, MPI_DOUBLE, &a.values[0], helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if(func != SUB)
            MPI_Scatterv(&b.values[0], counts, displs, MPI_DOUBLE, &b.values[0], helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        
    }
}

double distrDotProduct(Vector a, Vector b, int size, int me, int nprocs) {
    double dotProd = 0;

    sendVectors(a, b, helpSize, VV, me, size, nprocs);
    
    double temp = dotProduct(a, b, (nprocs - 1) * helpSize, size);

    MPI_Reduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    return dotProd;
    
}

Vector distrSubOp(Vector a, Vector b, int size, int me, int nprocs) {
    
    Vector res;
    Vector finalRes(size); 

    sendVectors(a, b, helpSize, SUB, me, size, nprocs);

    res = subtractVec(a, b,  (nprocs - 1) * helpSize, size);

    MPI_Gatherv(NULL, helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0)
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    

    return finalRes;
}

Vector distrSumOp(Vector a, Vector b, int size, int me, int nprocs) {
    
    Vector res;
    Vector finalRes(size); 

    sendVectors(a, b, helpSize, ADD, me, size, nprocs);

    res = addVec(a, b,  (nprocs - 1) * helpSize, size);


    MPI_Gatherv(NULL, helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0)
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    
    return finalRes;
}

Vector distrMatrixVec(CSR_Matrix A, Vector vec, int size, int me, int nprocs) {
    
    Vector finalRes(size);

    sendVectors(vec, Vector(0), helpSize, MV, me, size, nprocs);

    Vector res;
    res = sparseMatrixVector(A, vec, (nprocs - 1) * helpSize, size, size);

    MPI_Gatherv(NULL, helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if(res.size != 0){
        finalRes.values.insert(finalRes.values.begin() + ((nprocs-1)*helpSize), res.values.begin(), res.values.end());
    }

    return finalRes;
}