#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <unistd.h>

#define INF 1e9
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define COMM MPI_COMM_WORLD

#define send(buffer, count, dest) \
    MPI_Send(buffer, count, MPI_CHAR, dest, 0, COMM)

#define recv(buffer, count, src, status)\
    MPI_Recv(buffer, count, MPI_CHAR, src, MPI_ANY_TAG, COMM, &status)

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE  8
#define ROOT 0

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    wcout << "NBody.GPU" << endl << "=========" << endl << endl;

    //wcout << "CUDA version:   v" << CUDART_VERSION << endl;    
    //wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    wcout << "CUDA Devices: " << endl << endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;
    }
}

__global__ void floyd_warshall(int *d, int blockSize, int length, 
                            int XIndex, int YIndex, int kk) {
    int ii = blockSize * XIndex + blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockSize * YIndex + blockIdx.y * blockDim.y + threadIdx.y; 

    int dij = d[ii * length + jj];
    int dik = d[ii * length + kk];
    int dkj = d[kk * length + jj];
    
    if (dij > dik + dkj)
       d[ii * length + jj] = dik + dkj;   
}


int main(int argc, char **argv) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fprintf(stderr, "rank = %d, size = %d\n", rank, size);
     
    MPI_Status status;

    int blockSize = atoi(argv[3]);
     
    int N, M;
    int *edge;
    int *gpu[2];
    int gridSize;
    int length;
    const int deviceNum = 2;
    
    if (rank == ROOT) {
        FILE *fin = fopen(argv[1], "r");
        
        fscanf(fin, "%d %d", &N, &M);
        gridSize = N % blockSize ? N / blockSize + 1 : N / blockSize;
        length = blockSize * gridSize + 1;
        cudaMallocHost((void**)&edge, sizeof(int) * length * length);
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j)
                edge[i * length + j] = INF;
            edge[i * length + i] = 0;
        }
        while (M--) {
            int a, b, w;
            fscanf(fin, "%d %d %d", &a, &b, &w);
            a = a - 1;
            b = b - 1;
            edge[a * length + b] = w;
        }
        send(&N, sizeof(int), 1);
        send(&gridSize, sizeof(int), 1);
        send(&length, sizeof(int), 1);
        send(edge, sizeof(int) * length * length, 1);

        fclose(fin);
    } else {
        recv(&N, sizeof(int), ROOT, status);
        recv(&gridSize, sizeof(int), ROOT, status);
        recv(&length, sizeof(int), ROOT, status);
        cudaMallocHost((void**)&edge, sizeof(int) * length * length);
        recv(edge, sizeof(int) * length * length, ROOT, status);
        fprintf(stderr, "RANK %d, length = %d, gridSize = %d\n", rank, length, gridSize);      
    }
    
    cudaSetDevice(rank);
    cudaMalloc((void**)&gpu[rank], sizeof(int) * length * length);
    cudaMemcpy(gpu[rank], edge, sizeof(int) * length * length, H2D);
    cudaCheckErrors("memcpy error");   
    
    size_t sharedSize = 8 * 8;
    int blockNum = (N + blockSize - 1) / blockSize;
    int blockDimension = 8; 
    gridSize = blockSize / blockDimension;
    int gridFactor = 1024 / blockSize ;
    gridFactor *= gridFactor;

    dim3 blocks(gridSize, gridSize);
    dim3 threads(blockDimension, blockDimension);
    
    dim3 blockCol(gridSize * gridFactor, gridSize);
    dim3 blockRow(gridSize, gridSize * gridFactor);
    int remainBegin = (blockNum / gridFactor) * gridFactor ;  
    int remain = blockNum - remainBegin;
    dim3 blockColRemain(gridSize * remain, gridSize);
    dim3 blockRowRemain(gridSize, gridSize * remain);
    
    fprintf(stderr, "rank %d, gpu = %p \n", rank, gpu[rank]);   
/*
    if (rank == 0) { 
        send(&gpu[0], sizeof(int*), 1);
    }
    else {
        recv(&gpu[0], sizeof(int*), 0, status);
    }
    if (rank != 0) {
        fprintf(stderr, "gpu[0] = %p, gpu[1] = %p", gpu[0], gpu[1]); 
        floyd_warshall<<<blocks, threads>>>(gpu[0], blockSize, length, 0, 0, 0); 
        cudaCheckErrors("launch kernel");
     }
*/
/*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    int n = 5120 / 256;
    for (int i = 0; i < n; ++i) {
        if (rank == 0) {
            cudaMemcpy(edge, gpu[0], sizeof(int) * length * length, D2H);
            send(edge, sizeof(int) * length * length, 1);
            recv(edge, sizeof(int) * length * length, 1, status);
            cudaMemcpy(gpu[0], edge, sizeof(int) * length * length, H2D);
        } else {
            recv(edge, sizeof(int) * length * length, 0, status);
            cudaMemcpy(gpu[1], edge, sizeof(int) * length * length, H2D);
            cudaMemcpy(edge, gpu[1], sizeof(int) * length * length, D2H);
            send(edge, sizeof(int) * length * length, 0);
        }
        cudaCheckErrors("memcpy D2D");
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
*/
    for (int k = 0; k < blockNum; ++k) {
        //wcout << "rank = " << rank <<  "k = " << k  << ", blocknum = " << blockNum << endl;
        // phase one
        {
            for (int cur = 0; cur < blockSize; ++cur) {
                floyd_warshall<<<blocks, threads>>>
                    (gpu[rank], blockSize, length, k, k, k * blockSize + cur);
                cudaCheckErrors("phase one");
            }
        }
        // phase two
        {
            // Column 
            for (int i = 0; i < blockNum - remain; i = i + gridFactor) {
                for (int cur = 0; cur < blockSize; cur++) {
                    floyd_warshall<<<blockCol, threads>>>
                            (gpu[rank], blockSize, length, i, k, k * blockSize + cur);
                    cudaCheckErrors("phase two column main");
                }       
            }
        
            if (remainBegin < blockNum)
                for (int cur = 0; cur < blockSize; cur++) {
                    floyd_warshall<<<blockColRemain, threads>>>
                            (gpu[rank], blockSize, length, remainBegin, k, k * blockSize + cur);
                    cudaCheckErrors("phase two column remain");
                }
            // Row 
            for (int j = 0; j < blockNum - remain; j = j + gridFactor) {
                for (int cur = 0; cur < blockSize; ++cur) {
                    floyd_warshall<<<blockRow, threads>>>
                            (gpu[rank], blockSize, length, k, j, k * blockSize + cur);
                    cudaCheckErrors("phase two row main");
                }
            }
            if (remainBegin < blockNum)
                for (int cur = 0; cur < blockSize; cur++) {
                    floyd_warshall<<<blockRowRemain, threads>>>
                            (gpu[rank], blockSize, length, k, remainBegin, k * blockSize + cur);
                    cudaCheckErrors("phase two row remain");
                }
        }   
        
        //phase three
        {
            int thread = rank;
            int begin, end;
            cudaSetDevice(thread);
            
            if (thread == 0) {
                begin = 0;
                end = blockNum / 2;
            } else {
                begin = blockNum / 2;
                end = blockNum;
            }
            for (int i = begin; i < end; i++) {
                for (int j = 0; j < blockNum - remain; j = j + gridFactor) {
                    for (int cur = 0; cur < blockSize; ++cur) {
                        floyd_warshall<<<blockRow, threads>>>
                                (gpu[thread], blockSize, length, i, j, k * blockSize + cur);
                        cudaCheckErrors("phase three row main");
                    }
                }
                if (remainBegin < blockNum)
                    for (int cur = 0; cur < blockSize; cur++) {
                        floyd_warshall<<<blockRowRemain, threads>>>
                                (gpu[thread], blockSize, length, i, remainBegin, k * blockSize + cur);
                        cudaCheckErrors("phase three row remain");
                    }
            }    
        }
        int offset = (blockNum / 2) * blockSize * length ; 
        int copySize = length * length - offset;
        if (rank == 0) {
            cudaMemcpy(edge, gpu[0], sizeof(int) * offset, D2H);
            send(edge, sizeof(int) * offset, 1);
            recv(edge + offset, sizeof(int) * copySize, 1, status);
            cudaMemcpy(gpu[0] + offset, edge + offset, sizeof(int) * copySize, H2D);
            cudaCheckErrors("rank ROOT memcpy D2D");
        } else {
            recv(edge, sizeof(int) * offset, ROOT, status);
            cudaMemcpy(gpu[1], edge, sizeof(int) * offset, H2D);
            cudaMemcpy(edge + offset, gpu[1] + offset, sizeof(int) * copySize, D2H);
            cudaCheckErrors("rank 1 memcpy D2D");
            send(edge + offset, sizeof(int) * copySize, ROOT);
        }
    }
    if (rank == 0) {
        FILE *fout = fopen(argv[2], "w");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                if (edge[i * length + j] == INF)
                    fprintf(fout, "INF ");
                else
                    fprintf(fout, "%d ", edge[i * length + j]);
            }
            if (edge[i * length + N - 1] == INF)
                fprintf(fout, "INF\n");
            else
                fprintf(fout, "%d\n", edge[i * length + N - 1]);
       }
       fclose(fout);
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();   
    return 0;
}
