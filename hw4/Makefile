NVFLAGS  := -arch=sm_30
CXXFLAGS := -fopenmp
API = cuda
NAME = HW4_102062111_$(API)
SRC = $(NAME).cu

all: cuda
openmp: API = openmp
openmp: cuda
debug: NVFLAGS += -G -g
debug: cuda
sobel: NVFLAGS += -G -g

cuda:
	nvcc -o $(NAME) $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) $(SRC)

sobel: sobel.cu
	nvcc -o sobel $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) sobel.cu

clean:
	rm -f sobel prac
