.PHONY: all clean
CC = gcc
CFLAGS = -O3 -ffast-math -funroll-loops -fPIC -ftree-vectorize -march=native
LDFLAGS = -shared

all: lib_en.cpython-38-x86_64-linux-gnu.so

lib_en.cpython-38-x86_64-linux-gnu.so : en.o
	$(CC) $(LDFLAGS) $^ -o $@

en.o : en.c
	$(CC) $(CFLAGS) $^ -c

clean:
	rm *.so *.o
