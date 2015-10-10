#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <algorithm>

#define ROOT 0

int main(int argc, char *argv[])
{
    int rank, size;
    const int N = atoi(argv[1]);

    printf("Number of testcase = %d\n", N);

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    MPI_Status status;
    int *root_arr;
    MPI_File fin;
    int ret = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    
    if (rank == 0) {
        root_arr = new int[N+3];
        printf("Enter rank 0 statement ... \n");
        printf("File open status = %d\n" ,ret);
        MPI_File_read(fin, root_arr, N, MPI_INT, &status);
        
        for (int i = 0; i < N; ++i)
             printf("[Rank %d] root_arr[%d] = %d\n", rank, i, root_arr[i]); 
        MPI_File_close(&fin);
        printf("Out Rank 0 statement ... \n");
    } 
    
    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 
    int rank_num = size > N ? N : size;
    int num_per_node = N / rank_num;
    int *local_arr;
    int num_per_node_diff = N - num_per_node * size;
    if (num_per_node_diff > 0) {
        // Send remaining elements to size - 1
        int diff = num_per_node_diff; 
        if (rank == 0) {
            MPI_Send(root_arr + N - diff, diff, MPI_INT, size - 1, 0, MPI_COMM_WORLD); 
        } else if (rank == size - 1) {
            // Handle special case
            num_per_node += num_per_node_diff;
            local_arr = new int[num_per_node+1];
            MPI_Recv(local_arr + num_per_node - diff, diff, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    } else if(rank == rank_num - 1){
        local_arr = new int[num_per_node];
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 
    if (rank != rank_num - 1)
        local_arr = new int[num_per_node+1];
	
    // MPI_Scatter (send_buf, send_count, send_type, recv_buf, recv_count, recv_type, root, comm)
	MPI_Scatter(root_arr, num_per_node, MPI_INT, local_arr, num_per_node, MPI_INT, ROOT, MPI_COMM_WORLD);
    // printf("[Rank %d] num_per_node_size = %d\n" ,rank, num_per_node); 
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < num_per_node; ++i)
        if (rank != 0) 
            printf("[Rank %d] local_arr[%d] = %d\n", rank, i, local_arr[i]);
    
    if (rank == 0) {
        for (int i = 0; i < num_per_node; ++i)
             printf("[Rank %d] root_arr[%d] = %d\n", rank, i, local_arr[i]);
        delete [] root_arr;
    }
    delete [] local_arr; 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
     
    return 0;
}
