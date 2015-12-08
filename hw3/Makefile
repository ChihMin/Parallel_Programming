NAME = seq_MS
CC = g++
FLAG = -fopenmp
SRC = $(NAME).cpp
FLAGS = -fopenmp

debug: FLAGS += -g
debug: mpi_dynamic

mpi_static: CC = mpic++
mpi_static: NAME = MS_MPI_static
mpi_static: all

mpi_dynamic: CC = mpic++
mpi_dynamic: NAME = MS_MPI_dynamic
mpi_dynamic: all

openmp_dynamic: NAME = MS_OpenMP_dynamic
openmp_dynamic: all

openmp_static: NAME = MS_OpenMP_static
openmp_static: all

hybrid_static: CC = mpic++
hybrid_static: NAME = MS_Hybrid_static
hybrid_static: all

hybrid_dynamic: CC = mpic++
hybrid_dynamic: NAME = MS_Hybrid_dynamic
hybrid_dynamic: all

all:
	$(CC) $(FLAGS) $(SRC) -o $(NAME) -lX11

