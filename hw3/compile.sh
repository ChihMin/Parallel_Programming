#!/bin/bash

API="mpi openmp hybrid"
VER="static dynamic"

for i in $API; do
  for j in $VER; do
    make ${i}_$j
  done
done
