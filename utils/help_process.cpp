#include <iostream>
#include <mpi.h>

#include "headers/help_process.hpp"
#include "headers/mtx_ops_mkl.hpp"


/*
When the root node asks for help, this function is executed by all nodes (except the root node).
Each node receives the necessary vectors and executes the function func for a few values.
After the function is executed, the result is sent back to the root node.
*/
int helpProcess(const csr_matrix& A, int me, int size, int func, int * displs, int * counts) {
    double dotProd = 0;
    double temp = 0;
    double scalar = 0;
    dense_vector auxBuf(counts[me]);
    dense_vector auxBuf2(counts[me]);
    dense_vector auxBufMV(size);
    while(true) {
        MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if(func == ENDTAG)
            return 0;

        if(func == ADD || func == VV) {
            MPI_Scatterv(&auxBuf.values[0], counts, displs, MPI_DOUBLE, &auxBuf.values[0], counts[me], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
            MPI_Scatterv(&auxBuf2.values[0], counts, displs, MPI_DOUBLE, &auxBuf2.values[0], counts[me], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        }

        switch(func) {
            case MV:
                MPI_Bcast(&auxBufMV.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                sparseMatrixVector(A, auxBufMV, auxBuf2);
                MPI_Gatherv(&auxBuf2.values[0], counts[me], MPI_DOUBLE, &auxBuf2.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                break;

            case VV:
                dotProd = dotProduct(auxBuf, auxBuf2, counts[me]);
                MPI_Reduce(&dotProd, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                break;

            case ADD:
                MPI_Bcast(&scalar, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                addVec(auxBuf, auxBuf2, scalar, counts[me]);
                MPI_Gatherv(&auxBuf.values[0], counts[me], MPI_DOUBLE, &auxBuf.values[0], counts, displs, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                break;

            default:
                cout << "Process number: " << me << " Received wrong function tag from node " << ROOT << endl;
                return 1;
        }
    }
}