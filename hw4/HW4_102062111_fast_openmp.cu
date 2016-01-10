#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string.h>

#define INF 1e9
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE  8

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

__global__ void floyd_warshall_whole(int *d, int blockSize, int length, 
                            int XIndex, int YIndex, int ZIndex) {
    int ii = blockSize * XIndex + blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockSize * YIndex + blockIdx.y * blockDim.y + threadIdx.y; 
    //__shared__ int dij[8][8];
    
    // dij[threadIdx.x][threadIdx.y] = d[ii * length + jj];
    for (int cur = 0; cur < blockSize; ++cur) {
        int kk = ZIndex + cur;
        int dij = d[ii * length + jj]; 
        int dik = d[ii * length + kk];
        int dkj = d[kk * length + jj];
        
        if (dij > dik + dkj)
            d[ii * length + jj] = dik + dkj;      
/*
        if (dij[threadIdx.x][threadIdx.y] > dik + dkj)
           dij[threadIdx.x][threadIdx.y] = dik + dkj;
*/
    }
    // d[ii * length + jj] = dij[threadIdx.x][threadIdx.y];   
}


int main(int argc, char **argv) {
    
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    int blockSize = atoi(argv[3]);
     
    int N, M;
    int *edge;
    int *gpu[2];
    const int deviceNum = 2;

    fscanf(fin, "%d %d", &N, &M);
    int gridSize = N % blockSize ? N / blockSize + 1 : N / blockSize;
    int length = blockSize * gridSize + 1;
    edge = new int[length * length];
    
    //fprintf(stderr, "grid = %d, block = %d\n", gridSize, blockSize);

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
    
    
    for (int i = 0; i < deviceNum; ++i) {
        cudaSetDevice(i);
        cudaMalloc((void**)&gpu[i], sizeof(int) * length * length);
        cudaMemcpy(gpu[i], edge, sizeof(int) * length * length, H2D);
        cudaCheckErrors("melloc & copy gpu");
    }
    // Now only hangle N = 3200 testcase
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
    //wcout << "blocknum = " << blockNum << endl; 
    for (int k = 0; k < blockNum; ++k) {
        //wcout << "k = " << k << endl;
        // phase one
        cudaSetDevice(0);
        {
            for (int cur = 0; cur < blockSize; ++cur) {
                //wcout << "(" << cur << "/" << blockSize << endl;
                for (int id = 0; id < 2; ++id) {
                    cudaSetDevice(id);
                    floyd_warshall<<<blocks, threads>>>
                        (gpu[id], blockSize, length, k, k, k * blockSize + cur);
                }
                cudaCheckErrors("phase one");
            }
        }
        // phase two
        {
            // Column 
            for (int i = 0; i < blockNum - remain; i = i + gridFactor) {
                for (int cur = 0; cur < blockSize; cur++) {
                    for (int id = 0; id < deviceNum; ++id) {
                        cudaSetDevice(id);
                        floyd_warshall<<<blockCol, threads>>>
                                (gpu[id], blockSize, length, i, k, k * blockSize + cur);
                        
                        cudaCheckErrors("phase two column main");
                    }
                }       
            }
        
            if (remainBegin < blockNum)
                for (int cur = 0; cur < blockSize; cur++) {
                    for (int id = 0; id < deviceNum; ++id) {
                        cudaSetDevice(id);
                        floyd_warshall<<<blockColRemain, threads>>>
                                (gpu[id], blockSize, length, remainBegin, k, k * blockSize + cur);
                        cudaCheckErrors("phase two column remain");
                    }
                }
            // Row 
            for (int j = 0; j < blockNum - remain; j = j + gridFactor) {
                for (int cur = 0; cur < blockSize; ++cur) {
                    for (int id = 0; id < deviceNum; ++id) {
                        cudaSetDevice(id);
                        floyd_warshall<<<blockRow, threads>>>
                                (gpu[id], blockSize, length, k, j, k * blockSize + cur);
                        cudaCheckErrors("phase two row main");
                    }
                }
            }
            if (remainBegin < blockNum)
                for (int cur = 0; cur < blockSize; cur++) {
                    for (int id = 0; id < deviceNum; ++id) {
                        cudaSetDevice(id);
                        floyd_warshall<<<blockRowRemain, threads>>>
                                (gpu[id], blockSize, length, k, remainBegin, k * blockSize + cur);
                        cudaCheckErrors("phase two row remain");
                    }
                }
        }   
        
        //phase three
        {
            #pragma omp parallel num_threads(2)
            { 
                int thread = omp_get_thread_num();
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
                        int cur = 0;
                        floyd_warshall_whole<<<blockRow, threads>>>
                                (gpu[thread], blockSize, length, i, j, k * blockSize + cur);
                        cudaCheckErrors("phase three row main");
                    }
                    if (remainBegin < blockNum) {
                        int cur = 0;
                        floyd_warshall_whole<<<blockRowRemain, threads>>>
                                (gpu[thread], blockSize, length, i, remainBegin, k * blockSize + cur);
                        cudaCheckErrors("phase three row remain");
                    }
                }    
            }
            int offset = (blockNum / 2) * blockSize * length ; 
            int copySize = length * length - offset;
            cudaMemcpy(gpu[1], gpu[0], sizeof(int) * offset, D2D);
            cudaMemcpy(gpu[0] + offset, gpu[1] + offset, sizeof(int) * copySize, D2D);
        }
        
    }
    cudaSetDevice(0);
    cudaMemcpy(edge, gpu[1], sizeof(int) * length * length, D2H);
    
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
    return 0;
}
