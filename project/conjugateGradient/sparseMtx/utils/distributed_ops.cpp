#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "mtx_ops.cpp"

using namespace std;

#define ENDTAG 0
#define IDLETAG 1
#define FUNCTAG 2
#define MV 3
#define VV 4
#define SUB 6

void sendVectors(Vector a, Vector b, int begin, int helpSize, int dest, int func, int me, int size, int endRow) {

    MPI_Send(&helpSize, 1, MPI_DOUBLE, dest, IDLETAG, MPI_COMM_WORLD);
    MPI_Send(&func, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);
    MPI_Send(&helpSize, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);

    if(func == MV){
        MPI_Send(&a.values[0], size, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&begin, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&endRow, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
    }
    else {
        MPI_Send(&a.values[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        if(func == VV || func == SUB){
            MPI_Send(&b.values[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        }
    }
}
/*
double distrDotProduct(Sparse_Vec a, Sparse_Vec b, int size, int me, int nprocs) {
    
    int count = 0; 
    int flag = 0;
    int helpSize = size/nprocs;
    int end = 0;
    int dest = 1;

    while(count != nprocs - 1) {
        sendSDVectors(a, b, count * helpSize, helpSize, dest, VV, me, size, 0, 0);
        count++; dest++;
    }
    
    double dotProd = dotProductSparseVec(a, b, count * helpSize, size, me);

    double temp = 0;
    dest = 1;

    while(count > 0) {
        MPI_Recv(&temp, 1, MPI_DOUBLE, dest, VV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dotProd += temp;
        count--; dest++;
    }

    return dotProd;
    
}*/


Vector distrSubOp(Vector a, Vector b, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    int end = 0;
    int dest = 1;
    
    Vector res;
    Vector finalRes(size);  

    while(count != nprocs - 1) {
        sendVectors(a, b, count * helpSize, helpSize, dest, SUB, me, size, 0);
        count++; dest++;
    }

    res = subtractVec(a, b, count * helpSize, size);

    dest = 1;
    int index;
    for(index = 0; count > 0; index++) {
        MPI_Recv(&finalRes.values[index*helpSize], helpSize, MPI_DOUBLE, dest, SUB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dest++; count--;
    }

    if(res.size != 0)
        finalRes.values.insert(finalRes.values.begin() + (index*helpSize), res.values.begin(), res.values.end());
    

    return finalRes;
}


Vector distrMatrixVec(CSR_Matrix A, Vector vec, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    if(helpSize == 0) helpSize = 1; // caso extremo em que o numero de procs e maior que o tamanho do vetor
    int dest = 1;
    int index = 0;
    
    Vector finalRes(size);

    while(count != nprocs - 1 && count != size) {
        int begin = count * helpSize;
        int end = begin + helpSize;

        sendVectors(vec, Vector(0), begin, helpSize, dest, MV, me, size, end);
        count++; dest++;
    }
    Vector res;

    if(count <= size || count == 0)
        res = sparseMatrixVector(A, vec, count * helpSize, size, size);

    
    dest = 1;
    for(index = 0; count > 0; index++) {
        MPI_Recv(&finalRes.values[index*helpSize], helpSize, MPI_DOUBLE, dest, MV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dest++; count--;
    }
    
    if(index*helpSize != size){
        finalRes.values.insert(finalRes.values.begin() + (index*helpSize), res.values.begin(), res.values.end());
    }

    return finalRes;
}


void helpProccess(int helpDest, CSR_Matrix A, Vector b, int me, int size) {
    int func = -1;
    int helpSize = 0;
    double dotProd = 0;
    int begin = 0;
    int end = 0;

    MPI_Status status;
    MPI_Request sendIdleReq;
    
    Vector auxBuf(0);
    Vector auxBuf2(0);

    MPI_Recv(&func, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&helpSize, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    
    switch(func){
        case MV:
            auxBuf.resize(size);
            MPI_Recv(&auxBuf.values[0], size, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&begin, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&end, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);

            auxBuf2 = sparseMatrixVector(A, auxBuf, begin, end, size);
            MPI_Send(&auxBuf2.values[0], helpSize, MPI_DOUBLE, helpDest, MV, MPI_COMM_WORLD);
            break;
        /*case EVV:
            
            dotProd = dotProdSameVec(auxBuf, 0, me);

            MPI_Send(&dotProd, 1, MPI_DOUBLE, helpDest, EVV, MPI_COMM_WORLD);
            break;
        case VV:
            dotProd = dotProductSparseVec(auxBuf, auxBuf2, 0, helpSize, me);
            MPI_Send(&dotProd, 1, MPI_DOUBLE, helpDest, VV, MPI_COMM_WORLD);
            break;
            */
        case SUB:
            auxBuf.resize(helpSize);
            auxBuf2.resize(helpSize);
            MPI_Recv(&auxBuf.values[0], helpSize, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&auxBuf2.values[0], helpSize, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            auxBuf = subtractVec(auxBuf, auxBuf2, 0, helpSize);
            
            MPI_Send(&auxBuf.values[0], helpSize, MPI_DOUBLE, helpDest, SUB, MPI_COMM_WORLD);
            break;
        default: 
            cout << "Proccess number: " << me << " Received wrong function tag from node " << helpDest << endl;
            break;
    }
}