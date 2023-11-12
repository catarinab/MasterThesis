#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>


void helpProccess(CSR_Matrix A, int me, int size, int helpSize, int nprocs, int * displs, int * counts) {
    int func = -1;
    double dotProd = 0;
    int begin = 0;
    int end = 0;
    int savedVec = 0;
    int temp = 0;
    int helpDest = ROOT;

    MPI_Status status;
    MPI_Request sendIdleReq;
    
    DenseVector auxBuf(0);
    DenseVector auxBuf2(0);

    MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    
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
            auxBuf2 = sparseMatrixVector(A, auxBuf, displs[me], displs[me] + counts[me], size);
            MPI_Gatherv(&auxBuf2.values[0], counts[me], MPI_DOUBLE, &auxBuf2.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
            break;

        case VV:
            dotProd = dotProduct(auxBuf, auxBuf2, 0, counts[me]);
            MPI_Reduce(&dotProd, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            break;

        case SUB:
            auxBuf = subtractVec(auxBuf, auxBuf2, 0, counts[me]);
            MPI_Gatherv(&auxBuf.values[0], counts[me], MPI_DOUBLE, &auxBuf.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 
            break;

        case ADD:
            auxBuf = addVec(auxBuf, auxBuf2, 0, counts[me]);
            MPI_Gatherv(&auxBuf.values[0], counts[me], MPI_DOUBLE, &auxBuf.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 
            break;

        default: 
            cout << "Proccess number: " << me << " Received wrong function tag from node " << helpDest << endl;
            break;
    }
}