NVFLAGS  := -arch=sm_30
CXXFLAGS := -fopenmp
LIBS     := -I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -lmpich

all: sobel
debug: NVFLAGS += -G -g
debug: sobel

clean:
	rm -f sobel
sobel: sobel.cu
	nvcc -o sobel $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LIBS) sobel.cu
