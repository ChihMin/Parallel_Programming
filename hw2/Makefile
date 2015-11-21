CC = clang++
FLAGS = -Wno-int-to-pointer-cast -Wno-format
NAME = hw2_SRCC
SRC = $(NAME).cpp
TAR = ./$(NAME)

all: hw1
debug: FLAGS += -g
debug: hw1
debug: gdb 

hw1:
	$(CC) $(FLAGS) hw2_SRCC.cpp -o hw2_SRCC -lpthread

gdb:
	gdb -tui --args $(TAR)
