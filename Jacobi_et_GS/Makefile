CC = g++
CFLAGS = -Wall -Wextra -O3 -std=c++11 -march=native

all: jacobi_gs

jacobi_gs: main.cc methodes_J_GS.cc
	$(CC) $(CFLAGS) -o jacobi_gs main.cc methodes_J_GS.cc

clean:
	rm -f jacobi_gs
