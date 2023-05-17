#!/bin/bash
cores=$(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')
mpiCC main.cpp -I /usr/local/include/opencv4 -lopencv_core -lopencv_imgcodecs
mpirun -np "${1:-$cores}" a.out
rm output_ppm a.out img.ppm
