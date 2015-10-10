#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Please enter the correct format !"
  exit 0
fi

echo "number of n = $2"
clang++ read.cpp -o "read"
mpic++ -o prac prac.cpp
if [ $? != 0 ]; then
  echo "Compile ERROR, script stopping ... "
  exit 1
fi
./"read" $2 $3
mpirun  -np $1 ./prac $2 $3
