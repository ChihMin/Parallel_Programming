NAME = seq_MS
CC = g++
FLAG = -fopenmp
SRC = $(NAME).cpp
FLAGS = -fopenmp

debug: FLAGS += -g
debug: all

mpi_static: CC = mpic++
mpi_static: NAME = MS_MPI_static
mpi_static: all

openmp_dynamic: NAME = MS_OpenMP_dynamic
openmp_dynamic: all

openmp_static: NAME = MS_OpenMP_static
openmp_static: all


all:
	$(CC) $(FLAGS) $(SRC) -o $(NAME) -lX11

