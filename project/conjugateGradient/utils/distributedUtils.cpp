#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <bits/stdc++.h>

#include "utils.cpp"

using namespace std;

#define ENDTAG 0
#define IDLETAG 4
#define FUNCTAG 5
#define MV 6
#define VV 7
#define SUB 8

void sendVectors(vector<double> a, vector<double> b, int begin, int helpSize, int dest, int func, int me, int size) {
    MPI_Send(&helpSize, 1, MPI_DOUBLE, dest, IDLETAG, MPI_COMM_WORLD);
    MPI_Send(&func, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);
    MPI_Send(&helpSize, 1, MPI_INT, dest, FUNCTAG, MPI_COMM_WORLD);

    if(func == VV || func == SUB) {
        MPI_Send(&a[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&b[begin], helpSize, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
    }
    else if(func == MV)
        MPI_Send(&a[begin], size, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);

}


double distrDotProduct(vector<double> a, vector<double> b, int size, int me, int nprocs) {
    int count = 0; 
    int flag = 0;
    int helpSize = size/nprocs;
    int end = 0;
    int dest = 1;

    while(count != nprocs - 1) {
        sendVectors(a, b, count * helpSize, helpSize, dest, VV, me, size);
        count++;
        dest++;
    }
    
    double dotProd = dotProduct(a, b, count * helpSize, size);

    double temp = 0;
    dest = 1;

    while(count > 0) {
        MPI_Recv(&temp, 1, MPI_DOUBLE, dest, VV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dotProd += temp;
        count--; dest++;
    }

    return dotProd;
}

vector<double> distrSubOp(vector<double> a, vector<double> b, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    int end = 0;
    int dest = 1;
    
    vector<double> res;
    vector<double> finalRes(helpSize * (nprocs - 1));  

    while(count != nprocs - 1) {
        sendVectors(a, b, count * helpSize, helpSize, dest, SUB, me, size);
        count++; dest++;
    }

    if(helpSize*(nprocs-1) != size)
        res = subtractVec(a, b, count * helpSize, size);

    dest = 1;
    int index;
    for(index = 0; count > 0; index++) {
        MPI_Recv(&finalRes[index*helpSize], helpSize, MPI_DOUBLE, dest, SUB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dest++; count--;
    }
    
    if(res.size() != 0)
        finalRes.insert(finalRes.begin() + (index*helpSize), res.begin(), res.end());
    

    return finalRes;
}


vector<double> distrMatrixVec(vector<double> vec, vector<vector<double>> A, int size, int me, int nprocs) {
    int count = 0;
    int helpSize = size/nprocs;
    if(helpSize == 0) helpSize = 1; // caso extremo em que o numero de procs e maior que o tamanho do vetor
    int dest = 1;
    int index = 0;
    
    vector<double> res;
    vector<double> finalRes(helpSize * (nprocs - 1));   

    while(count != nprocs - 1 && count != size) {
        int begin = count * helpSize;
        int sendEnd = count * helpSize + helpSize;
        if(sendEnd > size) sendEnd = size;

        sendVectors(vec, vector<double>(0), 0, helpSize, dest, MV, me, size);
        MPI_Send(&begin, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);
        MPI_Send(&sendEnd, 1, MPI_DOUBLE, dest, FUNCTAG, MPI_COMM_WORLD);

        count++; dest++;
    }

    if(count <= size || count == 0)
        res = matrixVector(A, vec, count * helpSize, size, size);

    
    dest = 1;
    for(index = 0; count > 0; index++) {
        MPI_Recv(&finalRes[index*helpSize], helpSize, MPI_DOUBLE, dest, MV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        dest++; count--;
    }
    
    if(res.size() != 0)
        finalRes.insert(finalRes.begin() + (index*helpSize), res.begin(), res.end());

    return finalRes;
}


void helpProccess(int helpDest, vector<vector<double>> A, vector<double> b, int me, int size) {
    int func = -1;
    int helpSize = 0;
    double dotProd = 0;
    int begin = 0;
    int end = 0;

    MPI_Status status;
    MPI_Request sendIdleReq;

    vector<double> op(size);
    vector<double> auxBuf(size);
    vector<double> auxBuf2(size);

    MPI_Recv(&func, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&helpSize, 1, MPI_INT, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&auxBuf[0], size, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    if(func == VV || func == SUB) 
        MPI_Recv(&auxBuf2[0], helpSize, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
    
    switch(func){
        case MV:
            MPI_Recv(&begin, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&end, 1, MPI_DOUBLE, helpDest, FUNCTAG, MPI_COMM_WORLD, &status);

            op = matrixVector(A, auxBuf, begin, end, size);

            MPI_Send(&op[0], helpSize, MPI_DOUBLE, helpDest, MV, MPI_COMM_WORLD);
            break;
        case VV:
            dotProd = dotProduct(auxBuf, auxBuf2, 0, helpSize);
            MPI_Send(&dotProd, 1, MPI_DOUBLE, helpDest, VV, MPI_COMM_WORLD);
            break;
        case SUB:
            op = subtractVec(auxBuf, auxBuf2, 0, helpSize);
            
            MPI_Send(&op[0], helpSize, MPI_DOUBLE, helpDest, SUB, MPI_COMM_WORLD);
            break;
        default: 
            cout << "Proccess number: " << me << " Received wrong function tag from node " << helpDest << endl;
            break;
    }
}