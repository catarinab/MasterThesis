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
    dense_vector auxVec1(counts[me]);
    dense_vector auxVec2(size);
    while(true) {
        MPI_Bcast(&func, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if(func == ENDTAG)
            return 0;

        if(func == ADD || func == VV) {
            MPI_Scatterv(&auxVec1.values[0], counts, displs, MPI_DOUBLE, &auxVec1.values[0], counts[me], MPI_DOUBLE,
                         ROOT, MPI_COMM_WORLD);
            MPI_Scatterv(&auxVec2.values[0], counts, displs, MPI_DOUBLE, &auxVec2.values[0], counts[me], MPI_DOUBLE,
                         ROOT, MPI_COMM_WORLD);
        }

        switch(func) {
            case MV:
                MPI_Bcast(&auxVec2.values[0], size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                sparseMatrixVector(A, auxVec2, auxVec2);
                MPI_Gatherv(&auxVec2.values[0], counts[me], MPI_DOUBLE, &auxVec2.values[0], counts, displs, MPI_DOUBLE,
                            ROOT, MPI_COMM_WORLD);
                break;

            case NORM:
                MPI_Scatterv(&auxVec1.values[0], counts, displs, MPI_DOUBLE, &auxVec1.values[0], counts[me], MPI_DOUBLE,
                             ROOT, MPI_COMM_WORLD);
                dotProd = dotProduct(auxVec1, auxVec1, counts[me]);
                MPI_Reduce(&dotProd, &temp, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
                break;

            case VV:
                dotProd = dotProduct(auxVec1, auxVec2, counts[me]);
                MPI_Reduce(&dotProd, &temp, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
                break;

            case ADD:
                MPI_Bcast(&scalar, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                addVec(auxVec1, auxVec2, scalar, counts[me]);
                MPI_Gatherv(&auxVec1.values[0], counts[me], MPI_DOUBLE, &auxVec1.values[0], counts, displs, MPI_DOUBLE,
                            ROOT, MPI_COMM_WORLD);
                break;

            default:
                cout << "Process number: " << me << " Received wrong function tag from node " << ROOT << endl;
                return 1;
        }
    }
}