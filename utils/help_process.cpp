#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "headers/help_process.hpp"
#include "headers/mtx_ops_mkl.hpp"


/*
When the root node asks for help, this function is executed by all nodes (except the root node).
Each node receives the necessary vectors and executes the function func for a few values.
After the function is executed, the result is sent back to the root node.
*/
void helpProcess(csr_matrix A, int me, int size, int func, int nprocs, int * displs, int * counts) {
    double dotProd = 0;
    int temp = 0;
    
    dense_vector auxBuf(0);
    dense_vector auxBuf2(0);
    
    if(func == MV) {
        
        auxBuf.resize(size);
        MPI_Bcast(&auxBuf.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
    else {
        auxBuf.resize(counts[me]);
        auxBuf2.resize(counts[me]);
        MPI_Scatterv(&auxBuf.values[0], counts, displs, MPI_DOUBLE, &auxBuf.values[0], counts[me], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Scatterv(&auxBuf2.values[0], counts, displs, MPI_DOUBLE, &auxBuf2.values[0], counts[me], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
    
    switch(func) {
        case MV:
            auxBuf2 = sparseMatrixVector(A, auxBuf);
            MPI_Gatherv(&auxBuf2.values[0], counts[me], MPI_DOUBLE, &auxBuf2.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
            break;

        case VV:
            dotProd = dotProduct(auxBuf, auxBuf2);
            MPI_Reduce(&dotProd, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            break;

        case SUB:
            auxBuf = addVec(auxBuf, auxBuf2*(-1));
            MPI_Gatherv(&auxBuf.values[0], counts[me], MPI_DOUBLE, &auxBuf.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 
            break;

        case ADD:
            auxBuf = addVec(auxBuf, auxBuf2);
            MPI_Gatherv(&auxBuf.values[0], counts[me], MPI_DOUBLE, &auxBuf.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 
            break;

        default: 
            cout << "Process number: " << me << " Received wrong function tag from node " << ROOT << endl;
            exit(1);
    }
}