#!/bin/bash

EXEC=./hw2_NB_pthread

make NB_pthread
if [ $? != 0 ]; then
 echo "compile error ..."
 exit 1
fi 

if [ $1 == 1 ]; then
   $EXEC 1  10000 1000000000 0.01 test1.txt 0 enable -1   -1   3   600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 2 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
   $EXEC 40 1  10000 3 test2.txt 0 enable -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 3 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
   $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
  # $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
if [ $1 == 4 ]; then
  # $EXEC 1  10000 1000000 0.01 test1.txt 0 enable -1   -1   3   600
  # $EXEC 40 1  10000   1    test2.txt 0 enable -0.3 -0.3 0.6 600
  # $EXEC 40 1 20000 1 test3.txt 0.2 enable -0.5 -0.5 1 500
   $EXEC 40 1 300000 1 test4.txt 0.5 enable -1 -1  2.5 500
fi
