#!/bin/bash

versions="OpenMp"
policies="static dynamic"

time ./MS_OpenMP_dynamic 10 -2 2 -2 2 800 800 enabl
time ./MS_OpenMP_static 10 -2 2 -2 2 800 800 enabl
