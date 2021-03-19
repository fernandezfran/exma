.PHONY: all clean
CC = gcc
CFLAGS = -O3 -ffast-math -funroll-loops -fPIC -ftree-vectorize -march=native
LDFLAGS = -shared

all: lib_cn.cpython-38-x86_64-linux-gnu.so

lib_cn.cpython-38-x86_64-linux-gnu.so : cn.o
	$(CC) $(LDFLAGS) $^ -o $@

cn.o : cn.c
	$(CC) $(CFLAGS) $^ -c

clean:
	rm *.so *.o
