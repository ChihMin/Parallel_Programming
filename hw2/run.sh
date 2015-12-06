#!/bin/bash


EXEC=./hw2_NB_openmp
ENABLE=$3 

if [ ${#2} != 0 ]; then
  if [ $2 == "pthread" ]; then
    make NB_pthread
    EXEC=./hw2_NB_pthread
  elif [ $2 == "openmp" ]; then
    make NB_openmp   
    EXEC=./hw2_NB_openmp
  elif [ $2 == "debug" ]; then
    make debug
    EXEC="gdb -tui --args ./hw2_NB_BHalgo"
  elif [ $2 == "TA" ]; then
    EXEC=./hw2/NB-approximate
  elif [ $2 == "BH" ]; then
    make NB_BHalgo
    EXEC=./hw2_NB_BHalgo
  elif [ $2 == "nogrid" ]; then
    make NB_BHalgo_no_grid
    EXEC=./hw2_NB_BHalgo
  fi
fi

if [ $1 == 1 ]; then
  time $EXEC 1 10000 100000000 0.001 test1.txt 1 $ENABLE -1 -1 3 600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 2 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
  time $EXEC 1 1 2000 1 test2.txt 0.2 $ENABLE -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 3 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
  time $EXEC 8 1 2000 1 test3.txt 1 $ENABLE -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 4 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
   time $EXEC 8 1 20000 1 test4.txt 0.5 $ENABLE -1 -1  2.5 500
fi
if [ $1 == 5 ]; then
  time $EXEC 1 10000 1000000 0.01 test1.txt 0 $ENABLE -1 -1 3 600

  time $EXEC 1 1 2000 1 test2.txt 0 $ENABLE -0.3 -0.3 0.6 600

  time $EXEC 10 1 200000 1 test3.txt 0.2 $ENABLE -0.5 -0.5 1 500

  time $EXEC 10 1 30000 1 test4.txt 1 $ENABLE -1 -1  2.5 500
fi

