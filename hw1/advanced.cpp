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

//    printf("Number of testcase = %d\n", N);

    MPI_Init (&argc, &argv);

    double start_time, end_time;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
 //   printf("My rank is %d \n", rank); 
    
    //start_time = MPI_Wtime();

    MPI_File fin, fout;
    MPI_Status status;
    int *root_arr;
    int max_arr_size = size > N ? size : N;
    int ret = MPI_File_open(MPI_COMM_WORLD, argv[2], 
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    
    if (rank == ROOT) {
        root_arr = new int[max_arr_size+3];
//        printf("Enter rank 0 statement ... \n");
        MPI_File_read(fin, root_arr, N, MPI_INT, &status);
/*        
        for (int i = 0; i < N; ++i)
             printf("[START] [Rank %d] root_arr[%d] = %d\n", rank, i, root_arr[i]); 
        printf("Out Rank 0 statement ... \n");
*/
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
            MPI_Recv(local_arr + num_per_node - diff, diff, 
                    MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    } else if(rank == rank_num - 1) {
        local_arr = new int[num_per_node+1];
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 
    if (rank != rank_num - 1)
        local_arr = new int[num_per_node+1];
	

    // MPI_Scatter (send_buf, send_count, send_type, recv_buf, recv_count, recv_type, root, comm)
	if (rank < LAST)
        MPI_Scatter(root_arr, num_per_node, MPI_INT, local_arr, 
                    num_per_node, MPI_INT, ROOT, MPI_COMM_WORLD);
    else
        MPI_Scatter(root_arr, num_per_node-diff, MPI_INT, local_arr, 
                    num_per_node-diff, MPI_INT, ROOT, MPI_COMM_WORLD);
    
   // printf("[Rank %d] num_per_node_size = %d\n" ,rank, num_per_node); 
    MPI_Barrier(MPI_COMM_WORLD);
/*
    for (int i = 0; i < num_per_node; ++i)
        printf("[BEFORE] [Rank %d] local_arr[%d] = %d\n", rank, i, local_arr[i]); 
*/
    if (rank < rank_num) {
        std::sort(local_arr, local_arr + num_per_node);
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
/*
    for (int i = 0; i < num_per_node; ++i)
        printf("[AFTER] [Rank %d] local_arr[%d] = %d\n", rank, i, local_arr[i]); 
*/    
//    printf("rank %d is arrived\n", rank);
    
    MPI_Barrier(MPI_COMM_WORLD); // Wait for rank0 to read file 
    
    int *recv_buf, *send_buf;
    int recv_len, send_len, success;
    if (rank_num > 1 && rank < rank_num) {
        if (rank == ROOT) {
            send_len = num_per_node;
            MPI_Send(&send_len, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&success, 1, MPI_INT, rank+1,
                        MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Send(local_arr, send_len, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&recv_len, 1, MPI_INT, rank-1, 
                        MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            success = 1;
            MPI_Send(&success, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
            
            send_len = recv_len + num_per_node;
            recv_buf = new int[recv_len];
            send_buf = new int[send_len];
            
  //          printf("RANK %d recv_len = %d SUCCESS\n", rank, recv_len);
            MPI_Recv(recv_buf, recv_len, MPI_INT, 
                        rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
//            printf("RANK %d complete recevice array  SUCCESS\n", rank);
            
            int i = 0, j = 0, cur = 0;
            while (i < recv_len && j < num_per_node) {
                // Do MERGE array 
                if (recv_buf[i] < local_arr[j]) {
                    send_buf[cur++] = recv_buf[i++];
                } else {
                    send_buf[cur++] = local_arr[j++];
                }
            }
            while (i < recv_len)
                send_buf[cur++] = recv_buf[i++];

            while (j < num_per_node)
                send_buf[cur++] = local_arr[j++];
 /*          
            for (int k = 0; k < cur; k++) {
                printf("[RANK %d] send_buf[%d] = %d\n", rank, k, send_buf[k]);

            }
 */           
            if(rank != LAST) { 
                MPI_Send(&send_len, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
 //               printf("RANK %d send_len SUCCESS\n", rank);
                MPI_Recv(&success, 1, MPI_INT, rank+1, 
                        MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                MPI_Send(send_buf, send_len, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
 //               printf("RANK %d complete sending  array  SUCCESS\n", rank);
            }

            if(rank != LAST) 
                delete [] send_buf;
            delete [] recv_buf;
        }
    }

    
 //  printf("rank %d is arrived\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, argv[3], 
        MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fout);
    
    if (rank == LAST) {
        if (rank == 0) send_buf = local_arr;
        MPI_File_write(fout, send_buf, N, MPI_INT, &status);
/*        
        for (int i = 0; i < N; ++i) {
            printf("[FINAL] [Rank %d] ans[%d] = %d\n", rank, i, send_buf[i]);
        }
*/
    }
    MPI_File_close(&fout);
    
//    printf("CLOSE rank %d is arrived\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank != 0) {
        delete []  local_arr;
     //   printf("[FREE] [RANK %d] SUCCESS FREE\n", rank);
    } else {
        delete [] root_arr;
        delete [] local_arr;;
    }
    MPI_Finalize();
     
    return 0;
}
