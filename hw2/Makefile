CC = g++
FLAGS = -Wno-int-to-pointer-cast -Wno-format -lpthread
NBFLAGS = -O3 -Wno-int-to-void-pointer-cast -Wno-format -fopenmp
MPFLAGS = -O3 -Wno-int-to-void-pointer-cast -Wno-format -fopenmp
NAME = hw2_SRCC
SRC = $(NAME).cpp
TAR = ./$(NAME)

all: SRCC
NB_pthread: NAME = hw2_NB_pthread
NB_openmp: NAME = hw2_NB_openmp
NB_openmp: CC = g++
NB_BHalgo: NAME = hw2_NB_BHalgo
debug: MPFLAGS += -g
debug: NAME = NB_BHalgo
debug: NB_BHalgo

SRCC:
	$(CC) $(FLAGS)  $(SRC) -o $(TAR) -lpthread

NB_pthread:
	$(CC) $(NBFLAGS) $(SRC) -o $(TAR) -lX11

NB_openmp:
	$(CC) $(MPFLAGS) $(SRC) -o $(TAR) -lX11

NB_BHalgo:
	$(CC) $(MPFLAGS) $(SRC) -o $(TAR) -lX11
