NVFLAGS  := -arch=sm_20
CXXFLAGS := -fopenmp 
API = cuda
NAME = HW4_102062111_$(API)
SRC = $(NAME).cu

all: cuda
fast: API = fast
fast: cuda


debug: NVFLAGS += -G -g
debug: openmp
openmp: API = openmp
openmp: cuda
gpu: API = gpu
gpu: cuda
sobel: NVFLAGS += -G -g
test: NAME = test
test: NVFLAGS += -G -g
test: cuda

cuda:
	nvcc -o $(NAME) $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) $(SRC)

sobel: sobel.cu
	nvcc -o sobel $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) sobel.cu

clean:
	rm -f *cuda *openmp *mpi
