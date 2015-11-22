CC = clang++
FLAGS = -Wno-int-to-pointer-cast -Wno-format -lpthread
NBFLAGS = -Wno-int-to-void-pointer-cast -Wno-format -lpthread -lX11
NAME = hw2_SRCC
SRC = $(NAME).cpp
TAR = ./$(NAME)

all: SRCC
NB_pthread: NAME = hw2_NB_pthread
debug: FLAGS += -g
debug: hw1
debug: gdb 

SRCC:
	$(CC) $(FLAGS)  $(SRC) -o $(TAR) -lpthread

NB_pthread:
	$(CC) $(NBFLAGS) $(SRC) -o $(TAR) -lpthread

gdb:
	gdb -tui --args $(TAR)
