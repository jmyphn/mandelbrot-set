CPP=g++

CFLAGS=-lm
COPTFLAGS=-funroll-loops -ffast-math -O3 -march=native -mtune=native
MPIFLAGS=-DMPI

NVCC=nvcc
NVCCFLAGS=-DCUDA -O3 -dlto -use_fast_math -Xptxas -dlcm=ca -Xcompiler "-funroll-loops -ffast-math -O3 -march=native -mtune=native"

PYTHON = python3

all: mpi gpu serial

mpi: build/mpi
gpu: build/gpu
serial: build/serial

build/mpi: include/common.cpp include/main.cpp  mpi/mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAAGS) $(COPTFLAGS)

build/gpu: include/common.cpp include/main.cpp  gpu/gpu.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/serial: include/common.cpp include/main.cpp  serial/serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

.PHONY: clean

clean:
	rm -f build/mpi build/gpu build/serial