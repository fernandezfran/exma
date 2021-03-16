.PHONY: all clean
CC = gcc
CFLAGS = -O3 -ffast-math -funroll-loops -march=native -ftree-vectorize
LDFLAGS = -shared

all: lib_rdf.cpython-38-x86_64-linux-gnu.so

lib_rdf.cpython-38-x86_64-linux-gnu.so : rdf.o
	$(CC) $(LDFLAGS) $^ -o $@

rdf.o : rdf.c
	$(CC) $(CFLAGS) $^ -c

clean:
	rm *.so *.o
