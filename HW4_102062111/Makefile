NVFLAGS  := -arch=sm_20
CXXFLAGS := -fopenmp 
API = cuda
NAME = HW4_102062111_$(API)
SRC = $(NAME).cu

MPI_COMPILE_FLAGS = -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi
MPI_LINK_FLAGS = -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl 
# MPI_LIBS = $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS)
MPI_LIBS = -I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -lmpich

all: cuda

mpi: API = mpi

fast: API = fast
fast: cuda

debug: NVFLAGS += -G -g
debug: openmp

fast_openmp: API = fast_openmp
fast_openmp: cuda

fast_mpi: API = fast_mpi

openmp: API = openmp
openmp: cuda

gpu: API = gpu
gpu: mpi

sobel: NVFLAGS += -G -g

test: NAME = test
test: NVFLAGS += -G -g
test: cuda

cuda:
	nvcc -o $(NAME) $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) $(SRC)

sobel: sobel.cu
	nvcc -o sobel $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) sobel.cu

mpi:
	nvcc -o $(NAME)  $(NVFLAGS) $(MPI_LIBS) $(SRC)

fast_mpi:
	nvcc -o $(NAME)  $(NVFLAGS) $(MPI_LIBS) $(SRC)

clean:
	rm -f *cuda *openmp *mpi
