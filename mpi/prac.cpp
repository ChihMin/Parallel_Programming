#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <algorithm>

#define ROOT 0

void swap(int &x, int &y) {
    int tmp = x;
    x = y;
    y = tmp;
}

int main(int argc, char *argv[])
{
    int rank, size;
    const int N = atoi(argv[1]);

    printf("Number of testcase = %d\n", N);

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    printf("My rank is %d \n", rank); 
    
    MPI_File fin;
    MPI_Status status;
    int *root_arr;
    int ret = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    
    if (rank == ROOT) {
        root_arr = new int[N+3];
        printf("Enter rank 0 statement ... \n");
        MPI_File_read(fin, root_arr, N, MPI_INT, &status);
        
        for (int i = 0; i < N; ++i)
             printf("[START] [Rank %d] root_arr[%d] = %d\n", rank, i, root_arr[i]); 
        printf("Out Rank 0 statement ... \n");
    } 
    MPI_File_close(&fin);
    
    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 
    
    int rank_num = size > N ? N : size;
    const int LAST = rank_num - 1;
    int num_per_node = N / rank_num;
    int *local_arr;
    int num_per_node_diff = N - num_per_node * rank_num;
    int diff = num_per_node_diff;
    bool has_remain = false;
    bool has_remain_rank = rank_num % 2 ? true : false;
    
    if (num_per_node_diff > 0) {
        // Send remaining elements to size - 1
        has_remain = true;
        if (rank == ROOT) {
            MPI_Send(root_arr + N - diff, diff, MPI_INT, LAST, 0, MPI_COMM_WORLD); 
        } else if (rank == LAST) {
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
	if (rank != LAST)
        MPI_Scatter(root_arr, num_per_node, MPI_INT, local_arr, num_per_node, MPI_INT, ROOT, MPI_COMM_WORLD);
    else
        MPI_Scatter(root_arr, num_per_node-diff, MPI_INT, local_arr, num_per_node-diff, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    printf("[Rank %d] num_per_node_size = %d\n" ,rank, num_per_node); 
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < num_per_node; ++i)
        printf("[BEFORE] [Rank %d] local_arr[%d] = %d\n", rank, i, local_arr[i]); 

    int round = N % 2 ? N+1 : N;
    for (int i = 0; i < round; ++i) {
        // bool need_send = (i & 1)^(num_per_node & 1);
         bool need_send = true;
        for (int j = i & 1; j < num_per_node; j+=2) {
            if (j+1 < num_per_node) {
                if (local_arr[j] > local_arr[j+1]) 
                    swap(local_arr[j], local_arr[j+1]);        
            } else if (j-1 >= 0) {
                if (local_arr[j-1] > local_arr[j]) 
                    swap(local_arr[j-1], local_arr[j]);
            }            
        }
        
        if (need_send) {
            int element;
            bool recv_side = (i & 1);
            if (recv_side) {
                if (rank != ROOT) {
                    /* Receive element */
                    MPI_Recv(&element, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);  
                    MPI_Send(local_arr, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD); 
                    if (element > local_arr[0])
                        swap(element, local_arr[0]);
                }
            } else {
                /* Send element */
                if (rank != LAST) {
                    element = local_arr[num_per_node-1];
                    MPI_Send(&element, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD); 
                    MPI_Recv(&element, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);  
                    if (element < local_arr[num_per_node-1])
                        swap(element, local_arr[num_per_node-1]);
                }
            }
/*
            if (rank == ROOT) {
                for (int  i = 0; i < num_per_node; ++i){
                    printf("%d ", local_arr[i]);
                } printf("\n");
            }
*/
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < num_per_node; ++i)
        printf("[AFTER] [Rank %d] local_arr[%d] = %d\n", rank, i, local_arr[i]); 
    
    printf("rank %d is arrived\n", rank);
    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 

    int *ans;
    if (rank == ROOT) 
        ans = new int[N+3];

    if (has_remain && rank == rank_num - 1) {
        MPI_Gather(local_arr, num_per_node - diff, MPI_INT, ans, num_per_node - diff, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Send(local_arr + num_per_node - diff, diff, MPI_INT, ROOT, 0, MPI_COMM_WORLD); 
    }
    else {
        MPI_Gather(local_arr, num_per_node, MPI_INT, ans, num_per_node, MPI_INT, ROOT, MPI_COMM_WORLD);
        if (has_remain && rank == ROOT)
            MPI_Recv(ans + N - diff, diff, MPI_INT, LAST, MPI_ANY_TAG, MPI_COMM_WORLD, &status);  
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
             printf("[FINAL] [Rank %d] ans[%d] = %d\n", rank, i, ans[i]);
        delete [] root_arr;
    }

    delete [] local_arr; 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
     
    return 0;
}
