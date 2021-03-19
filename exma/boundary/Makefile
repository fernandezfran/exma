.PHONY: all clean
CC = gcc
CFLAGS = -O3 -ffast-math -funroll-loops -fPIC -ftree-vectorize -march=native
LDFLAGS = -shared

all: lib_boundary.cpython-38-x86_64-linux-gnu.so

lib_boundary.cpython-38-x86_64-linux-gnu.so : boundary.o
	$(CC) $(LDFLAGS) $^ -o $@

boundary.o : boundary.c
	$(CC) $(CFLAGS) $^ -c

clean:
	rm *.so *.o
