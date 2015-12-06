NAME = seq_MS
CC = g++
FLAG = -fopenmp
SRC = $(NAME).cpp
FLAGS = -fopenmp

debug: FLAGS += -g
debug: all

all:
	$(CC) $(FLAGS) $(SRC) -o $(NAME) -lX11

