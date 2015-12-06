#!/bin/bash

file=basic

if [ $# -lt 4 ]; then
  echo "Please enter the correct format !"
  exit 0
fi

echo "number of n = $2"
# clang++ read.cpp -o "read"
mpic++ -o $file -O3 ${file}.cpp
if [ $? != 0 ]; then
  echo "Compile ERROR, script stopping ... "
  exit 1
fi
# ./"read" $2 $3
time mpiexec  -np $1 ./$file $2 $3 $4
