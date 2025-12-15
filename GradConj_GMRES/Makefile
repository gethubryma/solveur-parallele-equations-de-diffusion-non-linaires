CC = g++
CFLAGS = -Wall -Wextra -O3 -std=c++11 -march=native 

all: grad_gmres

grad_gmres: main.cc GC_GMRES.cc
	$(CC) $(CFLAGS) -o grad_gmres main.cc GC_GMRES.cc

clean:
	rm -f grad_gmres
