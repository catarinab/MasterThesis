#include <iostream>
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
        displs[i] = i * helpSize;
        counts[i] = helpSize;
    }
    counts[nprocs - 1] += size % nprocs;
}

//send necessary vectors to all processes


void sendVectors(dense_vector& a, dense_vector& b, int func, int size) {
    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD); //broadcast need for help in function func

    MPI_Scatterv(&a.values[0], counts, displs, MPI_DOUBLE, MPI_IN_PLACE, helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(&b.values[0], counts, displs, MPI_DOUBLE, MPI_IN_PLACE, helpSize, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

}

//distribute dot product through all nodes
double distrDotProduct(dense_vector& a, dense_vector& b, int size, int me) {
    double dotProd = 0;
    sendVectors(a, b, VV, size);
    double temp = dotProduct(a, b, counts[me]);
    MPI_Reduce(&temp, &dotProd, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    return dotProd;
    
}

//distribute sum between vectors through all nodes
void distrSumOp(dense_vector& a, dense_vector& b, double scalar, int size, int me) {
    sendVectors(a, b, ADD, size);
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    addVec(a, b, scalar, counts[me]);
    MPI_Gatherv(MPI_IN_PLACE, helpSize, MPI_DOUBLE, &a.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
}

//distribute matrix-vector multiplication through all nodes
void distrMatrixVec(const csr_matrix& A, dense_vector& vec, dense_vector& res, int size) {
    int func = MV;
    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&vec.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    sparseMatrixVector(A, vec, res);
    MPI_Gatherv(MPI_IN_PLACE, helpSize, MPI_DOUBLE, &res.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
}