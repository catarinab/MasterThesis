#include <iostream>
#include <utility>
#include <mpi.h>

#include "headers/distr_mtx_ops.hpp"
#include "headers/mtx_ops_mkl.hpp"

using namespace std;

/*
This file contains the functions that are used to distribute the operations of the matrix and vectors
to all processes. 
The functions are called from the main file (exponentialMatrix.cpp) by the root node. 
*/

int * displs;
int * counts;
int helpSize = 0;

//initialize variables for MPI_GatherV and MPI_ScatterV
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

//send necessary vectors to all processes
void sendVectors(dense_vector a, dense_vector b, int func, int size) {
    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD); //broadcast need for help in function func

    
    if(func == MV)
        MPI_Bcast(&a.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    else {
        MPI_Scatterv(&a.values[0], counts, displs, MPI_DOUBLE, MPI_IN_PLACE, helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Scatterv(&b.values[0], counts, displs, MPI_DOUBLE, MPI_IN_PLACE, helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
}

//distribute dot product through all nodes
double distrDotProduct(dense_vector a, const dense_vector& b, int size, int me, int nprocs) {
    double dotProd = 0;

    sendVectors(a, b, VV, size);

    a.size = counts[0];

    double temp = dotProduct(a, b);

    MPI_Reduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    a.size = size;

    return dotProd;
    
}

//distribute sum between vectors through all nodes
dense_vector distrSumOp(dense_vector a, const dense_vector& b, double scalar, int size, int me, int nprocs) {
    dense_vector finalRes(size); 

    sendVectors(a, b, ADD, size);

    MPI_Bcast(&scalar, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    a.size = counts[0];

    dense_vector res = addVec(a, b, scalar);

    MPI_Gatherv(&res.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    a.size = size;

    return finalRes;
}

//distribute matrix-vector multiplication through all nodes
dense_vector distrMatrixVec(csr_matrix A, const dense_vector& vec, int size, int me, int nprocs) {
    
    dense_vector finalRes(size);

    sendVectors(vec, dense_vector(0), MV, size);

    dense_vector res = sparseMatrixVector(std::move(A), vec);

    MPI_Gatherv(&res.values[0], helpSize, MPI_DOUBLE, &finalRes.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    return finalRes;
}