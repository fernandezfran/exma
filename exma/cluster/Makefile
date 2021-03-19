.PHONY: all clean
CC = gcc
CFLAGS = -O3 -ffast-math -funroll-loops -fPIC -ftree-vectorize -march=native
LDFLAGS = -shared

all: lib_cluster.cpython-38-x86_64-linux-gnu.so

lib_cluster.cpython-38-x86_64-linux-gnu.so : cluster.o
	$(CC) $(LDFLAGS) $^ -o $@

cluster.o : cluster.c
	$(CC) $(CFLAGS) $^ -c

clean:
	rm *.so *.o
