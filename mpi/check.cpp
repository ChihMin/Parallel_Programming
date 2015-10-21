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
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
  /*  
    int x[] = {1,2,3,4,5,6};
    if (rank == 0) {
        MPI_Send(x, 3, MPI_INT, 1, 0, MPI_COMM_WORLD); 
    } else if (rank == 1) {
        MPI_Status status;
        MPI_Recv(x+3, 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        for (int i = 0; i < 6; ++i)
            printf("[rank %d] = %d\n", rank, x[i]);
    }
   */
    int x = 1, y = 123;
    printf("[Orig rank %d] y = %d\n", rank, x);
    if (rank == 0)  x = 100;
    MPI_Bcast(&x, 1, MPI_INT, ROOT, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Aftr rank %d] x = %d\n", rank, x);
     
    MPI_Finalize();
    return 0;
}
