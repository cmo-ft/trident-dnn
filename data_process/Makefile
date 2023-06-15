# Compiler: g++ for C++
CC=g++

# Some flags for compilation
CFLAGS=-g -Wall

# Root-related library and include directory
ROOTDEP=`root-config --cflags` `root-config --libs`

# Main command
analysis:  generate_csv.cpp 
	g++  generate_csv.cpp `root-config --cflags` `root-config --libs` -g -Wall -o generate_csv
